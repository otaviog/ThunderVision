import os

def extract_elapsed():
    file = open('f.txt', 'r')
    l = file.readline()
    elap = float(l)
    file.close()
    return elap

def run_test(limg, rimg, name):
    for metric in ['ssd', 'xcorr', 'bt']:
        for opt in ['wta', 'dyp', 'semiglobal']:
            for disp in [33, 64, 128, 256]:
                compl_name = "%s_%d_%s_%s" % (name, disp, metric, opt) 
                os.system("./SimpleBM --gpu --inputl %s --inputr %s --disparity %d --%s --%s --output %s.png > f.txt"
                          % (limg, rimg, disp, metric, opt, compl_name))
                print("%s %f" % (compl_name, extract_elapsed()))

run_test('../../res/tsukuba512_L.png', '../../res/tsukuba512_R.png', 'Tsukuba')
run_test('../../res/venus512_L.png', '../../res/venus512_R.png', 'Venus')
run_test('../../res/Benchmark/Teddy/q_left.png', '../../res/Benchmark/Teddy/q_right.png', 'Teddy')
run_test('../../res/Benchmark/Cones/q_left.png', '../../res/Benchmark/Cones/q_right.png', 'Cones')

