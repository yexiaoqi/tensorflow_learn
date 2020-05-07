import numpy as np
import tensorflow as tf


# 参考https://zhuanlan.zhihu.com/p/37740860
# def softmax(x):
#     x = np.exp(x) / np.sum(np.exp(x))
#     return x
#
#
# def softmax_prevent_overflow(x):
#     x -= np.max(x)
#     x = np.exp(x) / np.sum(np.exp(x))
#     return x


class Softmax():
    def __init__(self):
        pass

    def forward(self, x):
        self.out = np.copy(x)
        self.out -= np.max(self.out)
        self.out = np.exp(self.out)
        s = np.sum(self.out)
        self.out = self.out / s
        return self.out

    # i=j时候的yi和i!=j时候的0，拼起来就是diag(Y)
    # 没看出来eta的作用
    # def backward(self, eta):
    def backward(self):
        dout = np.diag(self.out) - np.outer(self.out, self.out)
        # return np.dot(dout, eta)
        return dout


class SoftWithLoss():
    def __init__(self):
        self.loss = None  # 损失
        self.a = None  # softmax的输出
        self.y = None  # 监督输出(one-hot vector)

    def softmax_prevent_overflow(self, x):
        x -= np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
        return x

    def forward(self, x, y):
        self.y = y
        # self.a = softmax(x)
        self.a = self.softmax_prevent_overflow(x)  # 注意必须有self
        self.loss = np.mean(-np.sum(self.y * np.log(self.a)))
        # self.loss = cross_entropy_error(self.a, self.y)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        dx = (self.a - self.y) / batch_size  # 请注意反向传播时，需除以批的大小（batch_size）
        return dx

# if __name__ == '__main__':
# print(softmax([2,3]))
# print(softmax_prevent_overflow([2, 3]))
