CamerasDesc default {

            Camera Left {
                   intrinsic_transform = [
                                       [1 2.5 3.66]
                                       [4. -5.4 6.]
                                       [7. 8. +9.9]]
                   intrinsic_distortion = [1. 2. 3.3e14 4.1e-10 -5.3e+10]
            }

            Camera Right {
                   intrinsic_transform = [
                                       [1. 2. 3.]
                                       [4. 8. 6.]
                                       [7. 8. 9.]]
                   intrinsic_distortion = [5. 4. 3. 2. 1.]
            }
            fundamental = [[9. 8. 7.]
                           [6. 5. 4.]
                           [3. 2. 1.]]
            extrinsics = [[[1. 4. 9.]
                           [2. 5. 8.]
                           [3. 6. 7.]]
                          [9. 3. 1.]]
}