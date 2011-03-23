CamerasDesc default {
            Camera Left {
                   intrinsic_transform = [
                                       [1. 2. 3.]
                                       [4. 5. 6.]
                                       [7. 8. 9.]]
                   intrinsic_distortion = [1. 2. 3. 4. 5.]
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
            extrinsics = [[
                                   [1. 4. 9.]
                                   [2. 5. 8.]
                                   [3. 6. 7.]]
                                   [9. 3. 1.]]
}