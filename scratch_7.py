# import numpy as np
# import matplotlib.pyplot as plt
#
# def rot_delta(delta):
#     return np.array([[1., 0., 0.], [0., np.cos(delta), np.sin(delta)], [0, -np.sin(delta), np.cos(delta)]])
#
# def rot_alpha(alpha):
#     return np.array([[np.cos(alpha), 0., np.sin(alpha)], [0., 1., 0.], [-np.sin(alpha), 0., np.cos(alpha)]])
#
# def rot_beta(beta):
#     return np.array([[1., 0., 0.], [0., np.cos(beta), -np.sin(beta)], [0, np.sin(beta), np.cos(beta)]])
#
# r = 10
# pa_space = np.linspace(0, np.pi*2., 80)
# x = r*np.cos(pa_space)
# y = r*np.sin(pa_space)
#
# plt.figure()
# plt.plot(x, y)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
#
# alpha = 0.5
# beta = 1.
# delta = 0.
#
# transform = np.matmul(rot_beta(beta), (np.matmul(rot_alpha(alpha), rot_delta(delta))))
#
# def rot_transform(args):
#     return np.matmul(rot_beta(args[1]), (np.matmul(rot_alpha(args[0]), rot_delta(args[2]))))
#
# x1, y1, z1 = np.matmul(rot_transform([alpha, beta, delta]), np.array([x, y, np.zeros_like(x)]))
#
# plt.figure()
# plt.plot(x1, y1)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
#
# import scipy.optimize as opt
#
# def circle(angle, r):
#     return [r*np.cos(angle), r*np.sin(angle)]
#
# print(opt.curve_fit(circle(np.linspace(0, 2.*np.pi, 30), r), pa_space, [x, y], p0=9))
# # res = opt.minimize(lambda arg: sum(abs(np.matmul(rot_transform(arg), np.array([x, y, np.zeros_like(x)])) - np.array([x1, y1, z1]))), x0=[0.5, 0.5, 0.])
# # print(res)
#%%
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

data = fits.open('/media/mouse13/My Passport/corotation/manga/1237648721749279075.fits')

#%%
print(data[0].header['BANDS'])
print(data[0].header['BAND0'])
print(data[0].header['BAND1'])


#%%
print(np.shape(data[0].data[0]))

#%%
plt.figure()
plt.imshow(data[0].data[2])
plt.show()