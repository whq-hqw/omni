def compliment_dim(input, dim):
    assert len(input) <= dim, "length of input should not larger than dim."
    if len(input) == dim:
        return input
    else:
        repeat = dim // len(input)
        input = input * repeat + [input[_] for _ in range(dim - len(input) * repeat)]
        return input