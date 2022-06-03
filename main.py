import os
import logging

if __name__ == '__main__':
    mode = "dev"
    data_path = os.path.join(os.getcwd(), "data", f"{mode}.txt")
    print(data_path)
