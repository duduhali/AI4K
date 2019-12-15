lr = 1e-3
the_lr = 1e-2
lr_len = 2
while lr+(1e-9)<the_lr:
    the_lr *=0.1
    lr_len +=1

print(the_lr,lr_len)


