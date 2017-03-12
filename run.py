from sampler import Sampler
from timeit import default_timer as timer
import numpy as np
sampler = Sampler(z_dim = 20, c_dim = 3, scale = 8.0, net_size = 30)
z_dim = 20
z1 =  z = np.random.uniform(-1.0, 1.0, size=(1, z_dim)).astype(np.float32)
#z2 =  z = np.random.uniform(-1.0, 1.0, size=(1, z_dim)).astype(np.float32)
#sampler.save_anim_gif(z1,z2, "out",n_frame=2,duration2=0.5)

for i in range(0,100):
    print("Step: {}".format(i))
    image1 = sampler.train(z=z1, image_path="c:/git/NetDrawer/cppn-tensorflow/source/obammers.jpg")
    #sampler.show_image(image1)
    if (i%20==0):

        image = sampler.generate(z=z1)
        sampler.show_image(image)
image = sampler.generate(z=z1)
sampler.show_image(image)



# for i in range(0,100):
#     start = timer()
#
#     for idx in range(z1.__len__()):
#         z1[idx] -=0.01
#     image = sampler.generate(z1,x_dim=512, y_dim=512)
#
#     sampler.show_image(image)
#     end = timer()
#     print(end - start)