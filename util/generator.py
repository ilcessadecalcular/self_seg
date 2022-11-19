import torch


def generator(model,x,crop_size):
    _,t,_,_,_=x.size()
    # over_lap = t % crop_size
    iter = t // crop_size
    output = torch.zeros_like(x)
    for i in iter:
        start_slice = i * crop_size
        end_slice = (i+1) * crop_size
        input = x[:,start_slice:end_slice]
        output1 = model(input)
        output = torch.cat((output,output1),1)
    input = x[:,t-crop_size:,:,:,:]
    output1 = model(input)
    output = torch.cat((output, output1[:,end_slice:,:,:,:]),1)
    return output