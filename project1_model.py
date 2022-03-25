import argparse
import os.path
import ssl
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import tqdm

ssl._create_default_https_context = ssl._create_unverified_context


def timer(fn):
	def wrapper(*args, **kwargs):
		start = time.time()
		r = fn(*args, **kwargs)
		end = time.time()
		print(f"{fn.__name__} uses: {end - start:.4f}s")
		return r

	return wrapper


def arg_parser():
	optimizer_option = ['sgd', 'adam', 'adamw']
	scheduler_option = ['plateau', '']
	measurement_option = ['min', 'max']
	bool_option = ['true', 'True', 'False', 'false']
	parser = argparse.ArgumentParser(description='project1')

	# args for the network
	parser.add_argument('--epoch', '-e', type=int, default=20, metavar='', help='training epochs')
	parser.add_argument('--channel', '-c', type=int, default=64, metavar='', help='channels after the first conv layer')
	parser.add_argument('--resnet', '-r', default=(3, 3, 3), nargs='*', type=int, metavar='', help='network model of the resnet, default is ResNet20 with 64 filters')

	# args for the training
	parser.add_argument('--optimzer', '-o', default='adamw', choices=optimizer_option, metavar='', help='optimizer')
	parser.add_argument('--learning_rate', '-l', default=1e-3, metavar='', type=float, help='learning rate')
	parser.add_argument('--batch_size', '-b', default=128, metavar='', type=int, help='batch size')
	parser.add_argument('--scheduler', '-s', choices=scheduler_option, default='plateau', metavar='', help='learning rate scheduler, default is plateau')

	# args for the lr_scheduler(use plateau)
	parser.add_argument('--factor', type=float, default=0.5, metavar='', help="lr scheduler's factor")
	parser.add_argument('--patience', type=int, default=50, metavar='', help="lr scheduler's patience")
	parser.add_argument('--threshold', type=float, default=1e-2, metavar='', help="lr scheduler's threshold")
	parser.add_argument('--lr_limit', type=float, default=1e-4, metavar='', help="lr scheduler's lr limit")
	parser.add_argument('--measurement', '-m', choices=measurement_option, default='min', metavar='', help="lr scheduler's measurement")

	# args for data augmentation and regulization
	parser.add_argument('--crop', type=int, default=(32, 4), nargs=2, metavar='', help='data augmentation: crop, input crop size, padding size')
	parser.add_argument('--flip', type=float, default=0.5, metavar='', help='data augmentation: flip, input the probability for flipping')
	parser.add_argument('--cutout', type=int, default=(1, 4), nargs=2, metavar='', help='data augmentation: cutout, input holes, cutout size')
	parser.add_argument('--label_smooth', type=float, default=0.2, metavar='', help='regularization: label smoothing')
	parser.add_argument('--weight_decay', '-w', default=1e-2, metavar='', help='regularization: L2 penalty')

	# args for training setting
	parser.add_argument('--progress_bar', '-p', choices=bool_option, type=str, default=False, metavar='', help='show progress bar when training')
	parser.add_argument('--cuda', choices=bool_option, type=str, default=True, metavar='', help='use GPU for the training')
	parser.add_argument('--test', choices=bool_option, type=str, default=False, metavar='', help='take a test run, skip waiting time for training, check bugs in code')
	parser.add_argument('--plot', choices=bool_option, type=str, default=True, metavar='', help='plot training curves')
	parser.add_argument('--save', choices=bool_option, type=str, default=False, metavar='', help='save the trained model')

	args = parser.parse_args()
	if args.weight_decay != 'b':
		args.weight_decay = float(args.weight_decay)

	# bug: -p false sets the parameter True, only -p '' sets the parameter false
	if type(args.progress_bar) == str:
		args.progress_bar = True if args.progress_bar.lower() == 'true' else False
	if type(args.cuda) == str:
		args.cuda = True if args.cuda.lower() == 'true' else False
	if type(args.test) == str:
		args.test = True if args.test.lower() == 'true' else False
	if type(args.plot) == str:
		args.plot = True if args.plot.lower() == 'true' else False
	if type(args.save) == str:
		args.save = True if args.save.lower() == 'true' else False

	args.cuda = args.cuda and torch.cuda.is_available()
	return args


class BasicBlock(nn.Module):
	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, ci, block, num_blocks, num_classes=10):
		super(ResNet, self).__init__()
		self.in_planes = ci
		self.l = []  # store all residual layers
		self.conv1 = nn.Conv2d(3, ci, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(ci)

		for i, v in enumerate(num_blocks, 1):  # calculate and build the residual layer, not manually build
			s = 1 if i == 1 else 2
			l = self._make_layer(block, ci * 2 ** (i - 1), v, stride=s)
			self.l.append(l)
			exec(f"self.layer{i} = l")  # every layer has to be written in __init__ explicitly, cannot just store in a list(why? wierd)

		self.linear = nn.Linear(ci * 2 ** (i - 1), num_classes)  # fc layer

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(
				block(self.in_planes, planes, stride))
			self.in_planes = planes
		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		for layer in self.l:
			out = layer(out)
		out = F.avg_pool2d(out, out.size()[-1])
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out


class Cutout(object):
	"""
	no cutout in Pytorch, write a class with __call__ to imitate transforms in T
	"""
	def __init__(self, hole, length):
		"""
		:param hole: number of holes on the image
		:param length: size of the holes
		"""
		self.hole = hole
		self.length = length

	def __call__(self, img):
		"""
		make it callable to use it at data processing stage along with other transforms in T
		:param img: image before cutout
		:return: image after cutout
		"""
		c, h, w = img.size()
		mask = np.ones((c, h, w), np.float32)

		for _ in range(self.hole):
			hole_left = np.random.randint(0, w - self.length + 1)
			hole_top = np.random.randint(0, h - self.length + 1)
			mask[:, hole_top: hole_top + self.length, hole_left: hole_left + self.length] = 0

		mask = torch.from_numpy(mask)
		img = torch.mul(img, mask)
		return img


class Project1:
	def __init__(self, args):
		"""
		vars for training
		:param args: command line arguments
		"""
		self.args = args
		self.train_loader = self.test_loader = None
		self.net = self.loss = self.optimizer = self.scheduler = None
		self.train_loss = []
		self.test_loss = []
		self.test_acc = []
		self.lr = []
		self.trainable_parameters = 0

	def process_data(self):
		"""
		stage for normalization and data augmentation
		"""
		download = False if os.path.isdir('CIFAR10') else True
		train = tv.datasets.CIFAR10('./CIFAR10/', download=download, train=True)
		mean = (train.data / 255).mean(axis=(0, 1, 2))
		std = (train.data / 255).std(axis=(0, 1, 2))

		data_aug = [T.ToTensor(), T.Normalize(mean=mean, std=std)]
		if self.args.crop != [0, 0]:
			data_aug.append(T.RandomCrop(size=self.args.crop[0], padding=self.args.crop[1]))
		if self.args.flip != 0:
			data_aug.append(T.RandomHorizontalFlip(p=self.args.flip))
		if self.args.cutout != [0, 0]:
			data_aug.append(Cutout(hole=self.args.cutout[0], length=self.args.cutout[1]))

		data_aug = T.Compose(data_aug)

		train = tv.datasets.CIFAR10('./CIFAR10/', download=False, train=True, transform=data_aug)
		test = tv.datasets.CIFAR10('./CIFAR10/', download=False, train=False, transform=T.ToTensor())
		self.train_loader = torch.utils.data.DataLoader(train, batch_size=self.args.batch_size, shuffle=True)
		self.test_loader = torch.utils.data.DataLoader(test, batch_size=self.args.batch_size, shuffle=False)

	def create_network(self):
		"""
		stage for creating network, loss, optimizer
		"""
		self.net = ResNet(self.args.channel, BasicBlock, self.args.resnet)
		if self.args.cuda:
			self.net = self.net.cuda()

		self.loss = nn.CrossEntropyLoss(label_smoothing=self.args.label_smooth)

		if self.args.weight_decay == 'b':
			optimizer_table = {
				'sgd': optim.SGD(self.net.parameters(), lr=self.args.learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-5 * self.args.batch_size),
				'adam': optim.Adam(self.net.parameters(), lr=self.args.learning_rate, amsgrad=True, weight_decay=1e-5 * self.args.batch_size),
				'adamw': optim.AdamW(self.net.parameters(), lr=self.args.learning_rate, amsgrad=True, weight_decay=1e-5 * self.args.batch_size)
			}
		else:
			optimizer_table = {
				'sgd': optim.SGD(self.net.parameters(), lr=self.args.learning_rate, momentum=0.9, nesterov=True, weight_decay=self.args.weight_decay),
				'adam': optim.Adam(self.net.parameters(), lr=self.args.learning_rate, amsgrad=True, weight_decay=self.args.weight_decay),
				'adamw': optim.AdamW(self.net.parameters(), lr=self.args.learning_rate, amsgrad=True, weight_decay=self.args.weight_decay)
			}

		self.optimizer = optimizer_table[self.args.optimzer.lower()]

		if self.args.scheduler == 'plateau':
			self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
				self.optimizer,
				self.args.measurement,
				threshold=self.args.threshold,
				factor=self.args.factor,
				patience=self.args.patience,
				min_lr=self.args.lr_limit,
			)

	def train_model(self):
		"""
		stage for training model and print performance every epoch
		save the trained model at the end
		"""
		for e in range(1, self.args.epoch + 1):
			train_loss = 0.0
			test_loss = 0.0
			test_acc = 0
			if self.args.progress_bar:
				self.train_loader = tqdm.tqdm(self.train_loader)
				self.test_loader = tqdm.tqdm(self.test_loader)

			for i, data in enumerate(self.train_loader):
				images, labels = data
				if self.args.cuda:
					images = images.cuda()
					labels = labels.cuda()
				self.optimizer.zero_grad()
				predicted_output = self.net(images)
				fit = self.loss(predicted_output, labels)
				fit.backward()
				self.optimizer.step()
				train_loss += fit.item()
				if self.args.test:
					break

			for i, data in enumerate(self.test_loader):
				with torch.no_grad():
					images, labels = data
					if self.args.cuda:
						images = images.cuda()
						labels = labels.cuda()
					predicted_output = self.net(images)
					test_acc += sum(i == j for i, j in zip(torch.argmax(predicted_output, dim=1), labels))  # test_acc
					fit = self.loss(predicted_output, labels)
					test_loss += fit.item()
				if self.args.test:
					break

			train_loss = train_loss / len(self.train_loader)
			test_loss = test_loss / len(self.test_loader)
			test_acc = test_acc / 10000
			test_acc = test_acc.tolist()
			lr = self.optimizer.param_groups[0]['lr']

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)
			self.test_acc.append(test_acc)
			self.lr.append(lr)

			if self.args.scheduler == 'plateau':
				if self.args.measurement == 'min':
					self.scheduler.step(test_loss)
				else:
					self.scheduler.step(test_acc)

			print(f'epoch {e}, train loss {train_loss:.4}, test loss {test_loss:.4}, test acc {test_acc:.4}, lr {lr:.4}')

		if self.args.save:
			torch.save(self.net.state_dict(), f"{int(time.time())}.pt")

	def plot_result(self):
		"""
		stage for ploting curves
		"""
		line_w = 1
		dot_w = 4
		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()

		# ax1 for train loss and test loss(left y-axis)
		# ax2 for test acc(right y-axis)
		ax1.plot(range(1, self.args.epoch + 1), self.train_loss, 'b--', linewidth=line_w, label='train trror')
		ax1.plot(range(1, self.args.epoch + 1), self.test_loss, 'r--', linewidth=line_w, label='test trror')
		ax2.plot(range(1, self.args.epoch + 1), self.test_acc, 'g--', linewidth=line_w, label='test tccuracy')

		# plot vertical lines for lr changing
		vline = False
		for i, v in enumerate(self.lr):
			if i >= 1 and v != self.lr[i - 1]:
				if not vline:
					plt.axvline(i + 1, linestyle='--', linewidth=1, label='lr reduced')
				else:
					plt.axvline(i + 1, linestyle='--', linewidth=1)
				vline = True

		lines, labels = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax2.legend(lines + lines2, labels + labels2, loc='upper left')

		ax1.scatter(range(1, self.args.epoch + 1), self.train_loss, color='b', s=dot_w)
		ax1.scatter(range(1, self.args.epoch + 1), self.test_loss, color='r', s=dot_w)
		ax2.scatter(range(1, self.args.epoch + 1), self.test_acc, color='g', s=dot_w)

		# note the highest test acc
		best_acc = max(self.test_acc)
		best_acc_epoch = self.test_acc.index(max(self.test_acc))
		ax2.scatter(best_acc_epoch + 1, best_acc, color='#7D3C98', marker='x', s=10, linewidths=3)
		ax2.annotate(
			f"epoch = {best_acc_epoch}, max acc = {best_acc}",
			(best_acc_epoch + 1, best_acc + 0.003),
			ha='center',
			color='#7D3C98'
		)

		ax1.set_xlabel('epoch')
		ax1.set_ylabel('loss')
		ax2.set_ylabel('accuracy')

		# set some value to make the figure looks better if epoch is too large
		ax1.set_ylim([0, max(max(self.train_loss), max(self.test_loss)) + 1])
		ax2.set_ylim([min(self.test_acc) - 0.001, 0.95])
		plt.grid(True)
		plt.show()

	def check_parameters(self):
		"""
		check model's trainable parameters
		:return False if parameters > 5M
		"""
		self.trainable_parameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

		print(f"trainable parameters: {self.trainable_parameters}")
		if self.trainable_parameters > 5_000_000:
			print('adjust the network first')
			return False
		else:
			return True

	@timer
	def main(self):
		print("------------------------")
		print(self.args)

		self.create_network()
		if not self.check_parameters():
			return

		self.process_data()
		self.train_model()
		if self.args.plot:
			self.plot_result()


def project1_model():
	return ResNet(64, BasicBlock, [3, 3, 3])


if __name__ == "__main__":
	try:
		table = arg_parser()
		Project1(table).main()
	except Exception as e:
		print('-----error happens, training stops-----')
		print(str(e))
