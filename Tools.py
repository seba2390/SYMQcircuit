def represent_integer_with_bits(number: int, nr_bits: int) -> str:
    """
    Represent an integer using a specific number of bits.

    Args:
        number (int): The integer to be represented.
        nr_bits (int): The number of bits to use for representation.

    Returns:
        str: A binary string representing the integer with leading zeros if required.
    """
    # Convert the integer to a binary string and remove the '0b' prefix
    binary_string = bin(number)[2:]
    # If the binary string is shorter than n, pad it with leading zeros
    binary_string = binary_string.zfill(nr_bits)
    return binary_string
