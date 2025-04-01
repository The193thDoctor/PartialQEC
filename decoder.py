import stim
import sinter
import numpy as np


class ComiledLookupTableDecoder(sinter.CompiledDecoder):
    def __init__(self, *, dem: stim.DetectorErrorModel):

        # Process detector locations and group them by the first element of the coordinates
        coordinates = dem.get_detector_coordinates()
        self.num_d = dem.num_detectors
        temp_dict = [[]*self.num_d]
        self.total_groups = 0
        for detector, coord in coordinates:
            if len(coord) >= 1:
                temp_dict[round(coord[0])].append(detector)
                self.total_groups = max(self.total_groups, round(coord[1] + 1))
        self.dict = []
        for i in range(self.total_groups):
            self.dict.append(np.array(temp_dict[i], dtype=np.int32))
        # speedup: if detectors are continuous, then only record first and te last detector in each group

        # Process Errors

        dem_flat = dem.flattened()

class LookUpTableDecoder(sinter.Decoder):
    def decode_via_files(self, *args, **kwargs):
        return NotImplementedError() # Do not implement decoding via files, only implement via decoder
    def compare_decoder_for_dem(self, *, dem: stim.DetectorErrorModel):


