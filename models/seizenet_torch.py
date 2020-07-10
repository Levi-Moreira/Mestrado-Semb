from torch.nn import Sequential, Conv2d, BatchNorm2d, ELU, MaxPool2d, Dropout

model = Sequential()
model.add_module("conv_time", Conv2d(1, 20, (10, 1), stride=1, ))
model.add_module("conv_spat", Conv2d(20, 20, (1, 18), stride=(1, 1), bias=False, ))
model.add_module("bnorm_1", BatchNorm2d(20, momentum=0.1, affine=True, eps=1e-5, ))
model.add_module("conv_1_nonlin", ELU())
model.add_module("pool_1", MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))

model.add_module("drop_2", Dropout(p=0.5))
model.add_module("conv_2", Conv2d(20, 40, (10, 20), stride=(1, 1), bias=False, ))
model.add_module("bnorm_2", BatchNorm2d(20, momentum=0.1, affine=True, eps=1e-5, ))
model.add_module("conv_2_nonlin", ELU())
model.add_module("pool_2", MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))

model.add_module("drop_3", Dropout(p=0.5))
model.add_module("conv_3", Conv2d(40, 80, (10, 40), stride=(1, 1), bias=True, ))
model.add_module("bnorm_3", BatchNorm2d(20, momentum=0.1, affine=True, eps=1e-5, ))
model.add_module("conv_3_nonlin", ELU())
