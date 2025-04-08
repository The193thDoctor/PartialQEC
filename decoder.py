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
        self.num_d = dem.num_detectors
        self.num_l = dem.num_observables
        if self.num_l > 64:
            raise Exception("Lookup table decoder only supports up to 64 logical observables")
        temp_g_to_d = [[] for _ in range(self.num_d)]
        self.d_to_g = np.zeros(self.num_d, dtype=np.uint32)
        self.num_g = 0
        for detector, coord in sorted(coordinates.items()):  # ensure that the detectors are sorted when grouped
            if len(coord) >= 1:
                temp_g_to_d[round(coord[0])].append(detector)
                self.d_to_g[detector] = round(coord[0])
                self.num_g = max(self.num_g, round(coord[0] + 1))
        for i in range(self.num_g):
            list.sort(temp_g_to_d[i])  # wrote for safety, but should be sorted already
        self.d_to_g_loc = np.zeros(self.num_d, dtype=np.uint32)
        for i in range(self.num_g):
            for j in range(len(temp_g_to_d[i])):
                self.d_to_g_loc[temp_g_to_d[i][j]] = j
        self.g_to_d = aw.Array(temp_g_to_d)
        self.g_size = np.array([len(self.g_to_d[i]) for i in range(self.num_g)], dtype=np.uint32)
        # speedup: if detectors are continuous, then only record first and te last detector in each group
        self.is_continuous = True
        for i in range(self.num_g):
            if not utils.is_continuous(self.g_to_d[i]):
                self.is_continuous = False
                break
        if self.is_continuous:
            self.d_stt = np.zeros(self.num_g, dtype=np.uint32)
            self.d_ter = np.zeros(self.num_g, dtype=np.uint32)
            for i in range(self.num_g):
                self.d_stt[i] = self.g_to_d[i][0]
                self.d_ter[i] = self.g_to_d[i][-1]
        for i in range(self.num_g):
            if len(self.g_to_d[i]) > 32:
                raise Exception("Lookup table decoder only supports up to 32 detectors in a Group")

        # Process Errors
        # Init data structures
        # WARNING: Using [[]]*(2**n) creates a list with n references to the same list object, not n different lists
        # This is a bug in the original code - let's fix it by using list comprehensions properly

        print(f"DEBUG: Initializing temp_decode structures for {self.num_g} groups with sizes: {self.g_size}")
        temp_decode_d = [[[] for _ in range(2 ** self.g_size[i])] for i in range(self.num_g)]
        temp_decode_error = [[float(0) for _ in range(2 ** self.g_size[i])] for i in range(self.num_g)]
        temp_decode_l = [[np.uint64(0) for _ in range(2 ** self.g_size[i])] for i in range(self.num_g)]

        dem_flat = dem.flattened()
        for instruction in dem_flat:
            if instruction.type == "error":
                targets_full: List[List[stim.DemTarget]] = instruction.target_groups()
                if len(targets_full) > 0:
                    targets = targets_full[0]
                    print("targets_full", targets_full)
                    error: float = instruction.args_copy()[0]

                    # Process detector flips
                    d_targets = np.array([target.val for target in targets if target.is_relative_detector_id()],
                                         dtype=np.int32)
                    d_gs: NDArray(np.int32) = self.d_to_g[d_targets]
                    d_g_locs: NDArray(np.int32) = self.d_to_g_loc[d_targets]
                    if len(d_gs) > 0:  # if d_gs is empty then we want to skip because g_min and g_max calculation blows up
                        g_min = np.int32(
                            np.min(d_gs))  # type conversion just to get rid of (mistaken-by-checker) warnings
                        g_max = np.int32(np.max(d_gs))  # same as above
                        d_flips = np.zeros(g_max - g_min + 1, dtype=np.uint32)
                        for i in range(len(d_gs)):
                            d_flips[d_gs[i] - g_min] ^= (1 << d_g_locs[i])
                        key = np.int32(d_flips[0])
                        if error < temp_decode_error[g_min][key] - 1e-10:
                            continue
                        temp_decode_d[g_min][key] = d_flips[1:]
                        temp_decode_error[g_min][key] = error

                        # Process observable flips (put it inside if statement because it requires g_min
                        l_targets = np.array([target.val for target in targets if target.is_logical_observable_id()],
                                             dtype=np.int32)
                        l_flips = np.uint64(0)
                        for l_target in l_targets:
                            l_flips ^= (1 << np.uint64(l_target))
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

        # DEBUG: Print the final datastructures
        print("\n=== DEBUG: Final decoder datastructures ===")
        for g in range(self.num_g):
            print(f"\nGroup {g} (size: {self.g_size[g]}):")
            for key in range(2 ** self.g_size[g]):
                # Convert key to binary representation (small endian)
                key_binary = ''.join(str((key >> i) & 1) for i in range(self.g_size[g]))

                # Get detector flips (d) as binary string
                d_flips = []
                for i in range(len(self.decode_d[g][key])):
                    val = self.decode_d[g][key][i]
                    if g + 1 + i < self.num_g:
                        binary = ''.join(str((val >> j) & 1) for j in range(self.g_size[g + 1 + i]))
                        d_flips.append(binary)

                # Get logical (l) flips as binary string
                l_val = self.decode_l[g][key]
                l_binary = ''.join(str((l_val >> i) & 1) for i in range(self.num_l))

                # Output in a compact format
                print(f"  Key: {key_binary} → Error: {self.decode_error[g][key]:.6f}, L: {l_binary}, D: {d_flips}")
        print("=== End of debug output ===\n")

    def decode_shots_bit_packed(
            self,
            *,
            bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        # Reshuffle the (rows) of detectors to groups
        num_shots = bit_packed_detection_event_data.shape[0]
        detection_reshuffled = np.zeros((num_shots, self.num_d), dtype=np.uint32)
        for d in range(self.num_d):
            detection_reshuffled[:, self.d_to_g[d]] ^= (np.uint64(
                ((bit_packed_detection_event_data[:, d >> 3] >> (d & 7)) & 1))) << np.uint32(self.d_to_g_loc[d])

        # Decoding
        l_flips = np.zeros(num_shots, dtype=np.uint64)
        for g in range(self.num_g):
            key = detection_reshuffled[:, g]
            len_impact = len(self.decode_d[g][0])
            for i in range(len_impact):
                detection_reshuffled[:, g + 1 + i] ^= self.decode_d[g, key, i]
            l_flips ^= np.array(self.decode_l[g, key], dtype=np.uint64)

        # Debug output
        display_shots = min(10, num_shots)  # Display at most 10 shots
        print("\n=== DEBUG: Detection events and correction details ===")
        for shot in range(display_shots):
            print(f"\nShot {shot} detection events:")
            # Print original detection events
            det_binary = []
            for d in range(self.num_d):
                val = (bit_packed_detection_event_data[shot, d >> 3] >> (d & 7)) & 1
                det_binary.append(str(val))
            print(f"  Original detection events: {''.join(det_binary)}")

            # Print reshuffled by group
            print("  Reshuffled by group:")
            for g in range(self.num_g):
                group_value = detection_reshuffled[shot, g]
                group_binary = ''.join(str((group_value >> i) & 1) for i in range(self.g_size[g]))
                print(f"    Group {g}: {group_binary}")

            # Print logical correction
            l_binary = ''.join(str((l_flips[shot] >> i) & 1) for i in range(self.num_l))
            print(f"  Logical correction: {l_binary}")
        print("=== End of debug output ===\n")

        # Reassemble into uint8 packed bits with little endian bit order
        num_bytes = (self.num_l + 7) >> 3
        l_flips_le = l_flips.astype('<u8', copy=False)
        result = l_flips_le.view(np.uint8).reshape(num_shots, 8)
        return result[:, :num_bytes]


class LookUpTableDecoder(sinter.Decoder):
    def decode_via_files(self, *args, **kwargs):
        raise NotImplementedError()  # Do not implement decoding via files, only implement via decoder

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel):
        return CompiledLookupTableDecoder(dem=dem)