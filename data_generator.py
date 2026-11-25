import random
import csv

def generate_two_digit_numbers():
    """Generate two random two-digit numbers."""
    num1 = random.randint(10, 99)
    num2 = random.randint(10, 99)
    return num1, num2

def calculate_carry(num1, num2):
    """Calculate the carry in the ones placefor the addition of two two-digit numbers."""
    ones_place_sum = (num1 % 10) + (num2 % 10)
    carry = ones_place_sum // 10
    return carry

def generate_data_pair(max_pairs=100):
    """Generate a pair of addition problems (base, source), such that:
    - The carry situation differs between the two problems.
    - The source problem has a carry in the ones place, while the base problem does not."""

    data_pairs = []

    while len(data_pairs) < max_pairs:
        base_num1, base_num2 = generate_two_digit_numbers()
        source_num1, source_num2 = generate_two_digit_numbers()

        base_carry = calculate_carry(base_num1, base_num2)
        source_carry = calculate_carry(source_num1, source_num2)

        if base_carry == 0 and source_carry > 0:
            data_pairs.append(((base_num1, base_num2), (source_num1, source_num2)))

    return data_pairs

def expected_output(num1, num2):
    """Calculate the expected output of the addition of two two-digit numbers."""
    return num1 + num2

def expected_output_after_carry(num1, num2):
    """Calculate the expected output of the addition of two two-digit numbers if hypothetically there always is a carry in the ones place."""
    return num1 + num2 + 10  # Add 10 to simulate the carry in the tens place

def save_pairs_to_csv(data_pairs, filename="data_pairs.csv"):
    """Save the generated data pairs to a CSV file and expected output after considering hypthetical carry in base case."""

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Base Num1", "Base Num2", "Source Num1", "Source Num2", "Expected Base Output", "Expected Source Output", "Expected Output After Carry"])
        for (base_num1, base_num2), (source_num1, source_num2) in data_pairs:
            expected_base_output = expected_output(base_num1, base_num2)
            expected_source_output = expected_output(source_num1, source_num2)
            expected_base_output_after_carry = expected_output_after_carry(base_num1, base_num2)
            writer.writerow([base_num1, base_num2, source_num1, source_num2, expected_base_output, expected_source_output, expected_base_output_after_carry])

if __name__ == "__main__":
    data_pairs = generate_data_pair(max_pairs=100)
    save_pairs_to_csv(data_pairs, filename="data_pairs.csv")
    print("Data pairs generated and saved to data_pairs.csv")