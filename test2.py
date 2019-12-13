import numpy as np

crop_sz = 480
step = 40
thres_sz = 48

w,h = 2040,2080

h_space = np.arange(0, h - crop_sz + 1, step)
print(h_space)
if h - (h_space[-1] + crop_sz) > thres_sz:
    h_space = np.append(h_space, h - crop_sz)
w_space = np.arange(0, w - crop_sz + 1, step)
print(w_space)
if w - (w_space[-1] + crop_sz) > thres_sz:
    w_space = np.append(w_space, w - crop_sz)

print(w_space)
print(h_space)

