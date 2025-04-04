import stim
import sinter
import numpy as np
import utils
from typing import List, Tuple
import awkward as aw
from numpy.typing import NDArray

class CompiledLookupTableDecoder(sinter.CompiledDecoder):
    def __init__(self, *, dem: stim.DetectorErrorModel):
        # Process detector locations and group them by the first element of the coordinates
        coordinates = dem.get_detector_coordinates()
        print(coordinates)
        self.num_d = dem.num_detectors
        self.num_l = dem.num_observables
        if self.num_l > 64:
            raise Exception("Lookup table decoder only supports up to 64 logical observables")
        temp_g_to_d = [[]] * self.num_d
        self.d_to_g = np.zeros(self.num_d, dtype=np.int32)
        self.num_g = 0
        for detector, coord in sorted(coordinates.items()): # ensure that the detectors are sorted when grouped
            if len(coord) >= 1:
                temp_g_to_d[round(coord[0])].append(detector)
                self.d_to_g[detector] = self.num_g
                self.num_g = max(self.num_g, round(coord[0] + 1))
        for i in range(self.num_g):
            list.sort(temp_g_to_d[i]) # wrote for safety, but should be sorted already
        self.d_to_g_loc = np.zeros(self.num_d, dtype=np.int32)
        for i in range(self.num_g):
            for j in range(len(temp_g_to_d[i])):
                self.d_to_g_loc[temp_g_to_d[i][j]] = j
        self.g_to_d = aw.Array(temp_g_to_d)
        self.g_size = np.array([len(self.g_to_d[i]) for i in range(self.num_g)], dtype=np.int32)
        # speedup: if detectors are continuous, then only record first and te last detector in each group
        self.is_continuous = True
        for i in range(self.num_g):
            if not utils.is_continuous(self.g_to_d[i]):
                self.is_continuous = False
                break
        if self.is_continuous:
            self.d_stt = np.zeros(self.num_g, dtype=np.int32)
            self.d_ter = np.zeros(self.num_g, dtype=np.int32)
            for i in range(self.num_g):
                self.d_stt[i] = self.g_to_d[i][0]
                self.d_ter[i] = self.g_to_d[i][-1]
        for i in range(self.num_g):
            if len(self.g_to_d[i]) > 32:
                raise Exception("Lookup table decoder only supports up to 32 detectors in a Group")

        # Process Errors
        # Init data structures
        temp_decode_d = [[[]]*(2 ** self.g_size[i]) for i in range(self.num_g)]
        temp_decode_error = [[float(0)] * (2 ** self.g_size[i]) for i in range(self.num_g)]
        temp_decode_l = [[np.uint64(0)]*(2 ** self.g_size[i]) for i in range(self.num_g)]

        dem_flat = dem.flattened()
        for instruction in dem_flat:
            if instruction.type == "error":
                targets_full: List[List[stim.DemTarget]] = instruction.target_groups()
                if len(targets_full) > 0:
                    targets = targets_full[0]
                    error: float = instruction.args_copy()[0]

                    # Process detector flips
                    d_targets = np.array([target.val for target in targets if target.is_relative_detector_id()], dtype=np.int32)
                    d_gs: NDArray(np.int32) = self.d_to_g[d_targets]
                    d_g_locs: NDArray(np.int32) = self.d_to_g_loc[d_targets]
                    g_min = np.int32(np.min(d_gs)) # type conversion just to get rid of (mistaken-by-checker) warnings
                    g_max = np.int32(np.max(d_gs)) # same as above
                    d_flips = np.zeros(g_max - g_min + 1, dtype=np.uint32)
                    for d_target in d_targets:
                        d_flips[d_gs[d_target] - g_min] ^= (1 << d_g_locs[d_target])
                    key = np.int32(d_flips[0])
                    if error < temp_decode_error[g_min][key] - 1e-10:
                        break
                    temp_decode_d[g_min][key] = d_flips[1:]


                    # Process observable flips
                    l_targets = np.array([target.val for target in targets if target.is_logical_observable_id()], dtype=np.int32)
                    l_flips = np.uint64(0)
                    for l_target in l_targets:
                        l_flips ^= (1 << l_target)
                    temp_decode_l[g_min][key] = l_flips

        # Update data structures and pad arrays for vectorized operations
        # For each group, ensure all decode_d entries have the same length
        for i in range(self.num_g):
            # Find the maximum length needed for this group
            max_length = 0
            for key in range(2 ** self.g_size[i]):
                if hasattr(temp_decode_d[i][key], "__len__"):
                    max_length = max(max_length, len(temp_decode_d[i][key]))
            
            # Pad all entries in this group to the same length
            if max_length > 0:
                for key in range(2 ** self.g_size[i]):
                    # Simple padding with np.pad
                    if len(temp_decode_d[i][key]) < max_length:
                        temp_decode_d[i][key] = np.pad(
                            temp_decode_d[i][key], 
                            (0, max_length - len(temp_decode_d[i][key])), 
                            'constant'
                        )
        
        self.decode_d = aw.Array(temp_decode_d)
        self.decode_error = aw.Array(temp_decode_error)
        self.decode_l = aw.Array(temp_decode_l)

    def decode_shots_bit_packed(
            self,
            *,
            bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        # Reshuffle the (rows) of detectors to groups
        num_shots = bit_packed_detection_event_data.shape[0]
        detection_reshuffled = np.zeros((num_shots, self.num_d), dtype=np.uint32)
        for d in range(self.num_d):
            detection_reshuffled[:, self.d_to_g[d]] ^= ((bit_packed_detection_event_data[:, d>>3] >> (d & 7)) & 1) << self.d_to_g_loc[d]

        # Decoding
        l_flips = np.zeros(num_shots, dtype=np.uint64)
        for g in range(self.num_g):
            key = detection_reshuffled[:, g]
            len_impact = len(self.decode_error[g][0])
            for i in range(len_impact):
                detection_reshuffled[:,g+1+i] ^= self.decode_d[g, key, i]
            l_flips ^= self.decode_l[g, key]

        # Reassemble into uint8 packed bits with little endian bit order
        num_bytes = (self.num_l + 7) >> 3
        l_flips_le = l_flips.astype('<u8', copy=False)
        result = l_flips_le.view(np.uint8).reshape(num_shots, 8)
        return result[:, :num_bytes]


class LookUpTableDecoder(sinter.Decoder):
    def decode_via_files(self, *args, **kwargs):
        return NotImplementedError() # Do not implement decoding via files, only implement via decoder

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel):
        return CompiledLookupTableDecoder(dem=dem)
