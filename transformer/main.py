from components.transformer import transformer


def print_hi(name):
    sample_transformer = transformer(
        vocab_size=8192,
        num_layers=6,
        units=512,
        d_model=256,
        num_heads=8,
        dropout=0.3,
        name="sample_transformer")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
