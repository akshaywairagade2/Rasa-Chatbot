with open('text','rw+') as file:
    for line in file:
        if not line.isspace():
            file.write(line)