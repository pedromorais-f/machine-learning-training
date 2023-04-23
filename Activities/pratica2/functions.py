import random
import csv


def number_of_pairs():
    n = input("Write the numbers of pairs:")

    return int(n)


def random_numbers_generator():
    var = random.uniform(0, 1)

    return var


def file_path(extension):
    name_file = input("Put the file's name:")

    name_file_split = name_file.split(".")

    if extension != name_file_split[1]:
        print("ERROR: Wrong extension file\n")
        return

    return name_file


def write_random_numbers_in_txt(n):
    random_numbers_list = []

    i = 0
    while i < n:
        num1 = random_numbers_generator()
        num2 = random_numbers_generator()
        data_list = [num1, num2]
        random_numbers_list.append(data_list)
        i += 1

    file_name = file_path("txt")

    try:
        with open(file_name, "w", newline='') as f:
            for i in range(n):
                f.write(str(random_numbers_list[i][0]) + ";" + str(random_numbers_list[i][1]) + "\n")
            f.close()
    except FileNotFoundError as error:
        print(error)


def write_random_numbers_in_csv(n):
    random_numbers_list = []

    i = 0
    while i < n:
        num1 = random_numbers_generator()
        num2 = random_numbers_generator()
        data_list = [num1, num2]
        random_numbers_list.append(data_list)
        i += 1

    file_name = file_path("csv")

    try:
        with open(file_name, "w", newline='') as f:
            data_writer = csv.writer(f)

            for row in random_numbers_list:
                data_writer.writerow(row)
            f.close()
    except FileNotFoundError as error:
        print(error)


def write_random_numbers_in_txt_v2(n):
    random_numbers_list = []

    for i in range(n):
        num1 = i + 1
        num2 = random.uniform(0, 1)
        data_list = [num1, num2]
        random_numbers_list.append(data_list)

    file_name = file_path("txt")

    try:
        with open(file_name, "w", newline='') as f:
            for i in range(n):
                f.write(str(random_numbers_list[i][0]) + ";" + str(random_numbers_list[i][1]) + "\n")
            f.close()
    except FileNotFoundError as error:
        print(error)


def write_random_numbers_in_csv_v2(n):
    random_numbers_list = []

    for i in range(n):
        num1 = i + 1
        num2 = random_numbers_generator()
        data_list = [num1, num2]
        random_numbers_list.append(data_list)

    file_name = file_path("csv")

    try:
        with open(file_name, "w", newline='') as f:
            data_writer = csv.writer(f)

            for row in random_numbers_list:
                data_writer.writerow(row)
            f.close()
    except FileNotFoundError as error:
        print(error)


def read_csv():
    file_name = file_path("csv")

    list_x = []
    list_y = []

    try:
        with open(file_name, "r") as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                for i, item in enumerate(row):
                    if i % 2 == 0:
                        list_x.append(float(item))
                    else:
                        list_y.append(float(item))
    except FileNotFoundError as error:
        print(error)

    return list_x, list_y


def read_txt():
    file_name = file_path("txt")

    list_x = []
    list_y = []

    try:
        with open(file_name, "r") as f:
            file_lines = f.readlines()

            for file_line in file_lines:
                numbers_list = file_line.split(";")

                list_x.append(float(numbers_list[0]))
                list_y.append(float(numbers_list[1]))
    except FileNotFoundError as error:
        print(error)

    return list_x, list_y
