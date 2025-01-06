
from test_lenet import test_lenet
from test_alexnet import test_alexnet
from test_googlenet import test_googlenet
from test_vggnet import test_vggnet
from test_resnet import test_resnet
from test_xception import test_xception
from test_senet import test_senet

if __name__ == "__main__":
    test_lenet()
    test_alexnet()
    test_googlenet()
    test_vggnet()
    test_resnet()
    test_xception()
    test_senet()
    print("All tests passed successfully!")
