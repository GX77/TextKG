# -*- coding: utf-8 -*-
# @Time    : 2023/2/19 23:23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : register.py



class LooseRegister:
    def __init__(self, init_dict=None):
        self._dict = init_dict if init_dict is not None else {}

    def register(self, name, target):
        self._dict[name] = target

    def __getitem__(self, item: str):
        for k in self._dict.keys():
            if item.startswith(k):
                return self._dict[k]
        raise KeyError(f"Key {item} not found in {list(self._dict.keys())}")

    def __contains__(self, key):
        return key in self._dict


if __name__ == '__main__':
    LOOSE_REGISTER = LooseRegister()

    @LOOSE_REGISTER.register()
    def func_a():
        print("a")


    def show_b():
        print("b")

