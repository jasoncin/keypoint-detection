def l2_loss(input, target, mask, batch_size):
    # print(input.shape)
    # print(target.shape)
    # print(mask.shape)
    loss = (input - target) * mask
    loss = (loss * loss) / 2 / batch_size

    return loss.sum()
