from train import HER

ckpoint = '/home/mihai/PycharmProjects/HER-lightning/her-pl/3f9soxa5/checkpoints/epoch=123.ckpt'

model = HER.load_from_checkpoint(ckpoint)
print(model.high_model.critic)