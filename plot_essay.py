import matplotlib.pyplot as plt
import torchvision as tv
import torchvision.transforms as T
import os
import random
import re
import sys

from project1 import Cutout

download = False if os.path.isdir('CIFAR10') else True
train = tv.datasets.CIFAR10('./CIFAR10/', download=False, train=True)

p2t = T.PILToTensor()
t2p = T.ToPILImage()


def f(img):
	plt.imshow(img)
	plt.axis('off')


def f_cutout(img, hole, size):
	cutout = Cutout(hole, size)
	img = cutout(p2t(img))
	plt.imshow((img / 255).permute(1, 2, 0))
	plt.axis('off')


def f_crop(img, padding, size):
	crop = T.RandomCrop(size=size, padding=padding)
	img = crop(img)
	plt.imshow(img)
	plt.axis('off')


def f_flip(img):
	flip = T.RandomHorizontalFlip(1)
	img = flip(img)
	plt.imshow(img)
	plt.axis('off')


def ff1(img):
	a = [(0, 0), (1, 4), (2, 4), (1, 16)]
	for i, (hole, size) in enumerate(a, 1):
		plt.subplot(2, 2, i)
		if hole == size == 0:
			f(img)
		else:
			f_cutout(img, hole, size)
	plt.savefig("cutout.png", bbox_inches=0)
	plt.show()


def ff2(img):
	for i in range(1, 5):
		plt.subplot(2, 2, i)
		if i == 1:
			f(img)
		else:
			f_crop(img, 4, 32)
	plt.savefig("crop.png", bbox_inches=0)
	plt.show()


def ff3():
	for i in range(1, 5):
		plt.subplot(2, 4, i)
		num = random.randint(0, 49999)
		f(train[num][0])
		plt.subplot(2, 4, i + 4)
		f_flip(train[num][0])

	plt.savefig("flip.png", bbox_inches=0)
	plt.show()


def main1():
	ff1(train[random.randint(0, 49999)][0])
	ff2(train[random.randint(0, 49999)][0])
	ff3()


def parse_txt(path):
	train_loss = []
	test_loss = []
	test_acc = []
	lr = []
	with open(path, 'r') as f:
		lines = f.readlines()

	p = re.compile(r'train loss (.*), test loss (.*), test acc (.*), lr (.*)\n')
	for l in lines:
		a = re.search(p, l)
		if not a:
			continue
		a, b, c, d = a.groups()
		train_loss.append(float(a))
		test_loss.append(float(b))
		test_acc.append(float(c))
		lr.append(float(d))

	return train_loss, test_loss, test_acc, lr


def compare_curve(result_list, label, dot=True):
	line_w = 1.5
	dot_w = 4
	e = len(result_list[0])

	for result, l in zip(result_list, label):
		plt.plot(range(1, e + 1), result, linewidth=line_w, label=l)
		if dot:
			plt.scatter(range(1, e + 1), result, s=dot_w)
	plt.legend()
	plt.grid(True)
	return plt


def compare_structure():
	_, loss20, acc20, _ = parse_txt('results/slurm-16525477.out')
	_, loss32, acc32, _ = parse_txt('results/slurm-16525534.out')
	_, loss44, acc44, _ = parse_txt('results/slurm-16525535.out')
	_, loss56, acc56, _ = parse_txt('results/slurm-16525546.out')
	_, loss110, acc110, _ = parse_txt('results/slurm-16525553.out')
	label = [
		'resnet20 + 64filter',
		'resnet32 + 32fillter',
		'resnet44 + 32fillter',
		'resnet56 + 32fillter',
		'resnet110 + 16fillter',
	]
	plt.style.use('ggplot')
	plt.figure(figsize=(12, 6), dpi=80)
	plt.subplot(1, 2, 1)
	p = compare_curve([loss20, loss32, loss44, loss56, loss110], label)
	p.xlabel('epoch')
	p.ylabel('test loss')

	plt.subplot(1, 2, 2)
	p = compare_curve([acc20, acc32, acc44, acc56, acc110], label)
	p.xlabel('epoch')
	p.ylabel('test accuracy')

	plt.savefig('compare_structure.png')
	plt.savefig('report/compare_structure.png')
	plt.show()


def compare_aug():
	_, no_opt, acc1, _ = parse_txt('results/no_opt.txt')
	_, crop, acc2, _ = parse_txt('results/crop.txt')
	_, cutout, acc3, _ = parse_txt('results/cutout.txt')
	_, flip, acc4, _ = parse_txt('results/flip.txt')
	_, best, acc5, _ = parse_txt('results/all.txt')

	label = [
		'no augmentation',
		'crop(32, 4)',
		'cutout(1, 4)',
		'flip(0.5)',
		'crop(32, 4) + cutout(1, 4) + flip(0.5)',
	]
	plt.style.use('ggplot')
	plt.figure(figsize=(12, 6), dpi=80)
	plt.subplot(1, 2, 1)
	p = compare_curve([no_opt, crop, cutout, flip, best], label)
	p.xlabel('epoch')
	p.ylabel('test loss')

	plt.subplot(1, 2, 2)
	p = compare_curve([acc1, acc2, acc3, acc4, acc5], label)
	p.xlabel('epoch')
	p.ylabel('test accuracy')

	plt.savefig('compare_aug.png')
	plt.savefig('report/compare_aug.png')
	plt.show()


def compare_reg():
	_, _, no_reg, _ = parse_txt('results/no_reg.txt')
	_, _, label_s, _ = parse_txt('results/label.txt')
	_, _, weight, _ = parse_txt('results/weight.txt')
	_, _, weight_batchsize, _ = parse_txt('results/weight_batchsize.txt')
	_, _, best, _ = parse_txt('results/all.txt')

	label = [
		'no regularization',
		'label smoothing($\epsilon=0.2$)',
		'weight_decay = 0.2',
		'weight_decay = 1e-5*batch size',
		'label smoothing($\epsilon=0.2$), weight_decay = 0.2'
	]
	plt.style.use('ggplot')
	p = compare_curve([no_reg, label_s, weight, weight_batchsize, best], label)
	p.xlabel('epoch')
	p.ylabel('test accuracy')
	p.savefig('compare_reg.png')
	p.savefig('report/compare_reg.png')
	p.show()


def compare_optimzier():
	_, sgd_loss, sgd_acc, _ = parse_txt('results/sgd.txt')
	_, adam_loss, adam_acc, _ = parse_txt('results/adam.txt')
	_, adamw_loss, adamw_acc, _ = parse_txt('results/all.txt')

	label = [
		'SGD',
		'Adam',
		'AdamW',
	]
	plt.style.use('ggplot')
	plt.figure(figsize=(12, 6), dpi=80)
	plt.subplot(1, 2, 1)
	p = compare_curve([sgd_loss, adam_loss, adamw_loss], label)
	p.xlabel('epoch')
	p.ylabel('test loss')

	plt.subplot(1, 2, 2)
	p = compare_curve([sgd_acc, adam_acc, adamw_acc], label)
	p.xlabel('epoch')
	p.ylabel('test accuracy')

	plt.savefig('compare_optimizer.png')
	plt.savefig('report/compare_optimizer.png')
	plt.show()


def compare_scheduler():
	dot = False
	_, loss1, acc_1, _ = parse_txt('results/slurm-16597410.out')
	_, loss2, acc_2, _ = parse_txt('results/slurm-16597294.out')
	_, loss3, acc_3, _ = parse_txt('results/slurm-16597715.out')
	_, loss4, acc_4, _ = parse_txt('results/slurm-16597719.out')
	_, loss5, acc_5, _ = parse_txt('results/slurm-16624156.out')

	label = [
		'patience=10, factor=0.1, threshold=1e-2',
		'patience=10, factor=0.1, threshold=1e-4',
		'patience=50, factor=0.5, threshold=1e-2',
		'patience=50, factor=0.5, threshold=1e-4',
		'no learning rate scheduler'
	]
	plt.style.use('ggplot')
	plt.figure(figsize=(18, 6), dpi=80)
	plt.subplot(1, 2, 1)
	p = compare_curve([loss1, loss2, loss3, loss4, loss5], label, dot)
	p.xlabel('epoch')
	p.ylabel('test loss')
	p.ylim([0.99, 1.3])

	plt.subplot(1, 2, 2)
	p = compare_curve([acc_1, acc_2, acc_3, acc_4, acc_5], label, dot)
	p.xlabel('epoch')
	p.ylabel('test accuracy')
	p.ylim([0.8, 0.95])

	plt.savefig('compare_scheduler.png')
	plt.savefig('report/compare_scheduler.png')
	plt.show()


if __name__ == "__main__":
	compare_structure()
	compare_aug()
	compare_reg()
	compare_optimzier()
	compare_scheduler()
