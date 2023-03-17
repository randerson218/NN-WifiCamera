data = []

with open('data.csv') as csvfile:
        data.append(csvfile.read().strip().replace("\n",",").split(","))
        
print(data)
