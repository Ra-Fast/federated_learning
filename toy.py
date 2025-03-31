import hydra
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

def function_test(x,y):
    result=x+y
    print(f"result: {result= }")
    return result

class MyClass:
    def __init__(self, x):
        self.x=x

    def print_x_squared(self):
        print(f"x squared: {self.x**2= }")
    
class MyComplexClass:
    def __init__(self, my_object: MyClass):
        self.obj=my_object

    def instantiate_object(self):
        self.obj=instantiate(self.obj)


@hydra.main(config_path="conf", config_name="toy", version_base=None)
def main(cfg: DictConfig):

    # Print the configuration
    print(OmegaConf.to_yaml(cfg))
    
    # easy
    print(cfg.foo)
    print(cfg.bar.baz)
    print(cfg.bar.more)
    print(cfg.bar.more.blabla)

    # less easy
    output=call(cfg.my_func)
    print(f"output: {output= }")

    output=call(cfg.my_func, y=100)
    print(f"output: {output= }")

    print("partials")
    fn=call(cfg.my_partial_func)

    output=fn(y=1000)

    print("Objects")
    obj=instantiate(cfg.my_object)
    obj.print_x_squared()

    print("--------"*10)
    complex_obj=instantiate(cfg.my_complex_object)
    #complex_obj.obj.print_x_squared()
    print(complex_obj.obj)
    complex_obj.instantiate_object()
    print(complex_obj.obj.print_x_squared())

    print("--------"*10)
    print(cfg.toy_model)
    mymodel=instantiate(cfg.toy_model)
    print(mymodel)


if __name__ == "__main__":
    main()