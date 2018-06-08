import csv

# соединяем все данные в один файл

data_file = open('data/data.csv', 'w')
data_writer = csv.writer(data_file, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

with open('data/telegram.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data_writer.writerow([row[1], row[2]])

with open('data/vk.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data_writer.writerow([row[1], row[3]])

with open('data/positive.csv') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=';')
    k = 0
    for row in csvreader:
        if(k==500):
            break
        data_writer.writerow([row[3], 'positive'])
        k += 1