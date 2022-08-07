import torch


def dice_func(output, target):
    smooth = 0.001
    intersection = output*target
    intersection_sum = torch.sum(intersection)
    print(intersection_sum)
    output_sum = output.sum()
    print(output_sum)
    target_sum = target.sum()
    print(target_sum)
    f = 2*intersection_sum / (output_sum + target_sum +smooth)

    return f
