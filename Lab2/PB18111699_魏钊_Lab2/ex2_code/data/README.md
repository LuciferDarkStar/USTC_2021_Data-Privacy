# Data

MNIST & CIFAR-10 datasets will be downloaded automatically by the torchvision package.


(base) C:\Users\Lucifer.dark\Desktop\ex2_v2\ex2_code>python main.py --mode Paillier  --num_users 1 --gpu 0
cuda:0
load dataset...
clients and server initialization...
start training...
C:\Python_Anaconda\lib\site-packages\torch\nn\functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not
use them for anything important until they are released as stable. (Triggered internally at  ..\c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
Round   0, Training average loss 0.266
Round   0, Testing accuracy: 12.19
训练时间s： 1188.805143589737
Round   1, Training average loss 0.188
Round   1, Testing accuracy: 12.19
训练时间s： 2368.7180514335632
