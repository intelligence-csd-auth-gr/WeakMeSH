import pickle


def create_seed_word_list(filename):
    with open(filename, mode='rb') as file:
        mesh = file.readlines()
    new_list=list()
    ok='false'
    dictionary=dict()
    count=0
    for line in mesh:
        if (str(line).__contains__("b'MH = ")):
            #print(line)
            heading=str(line).replace("b'MH = ","").strip("\n'")[:-2]
            ok='true'
            count = 0
            seed_words = ""
            description = ""
            previous_indexings=""
            tree=set()
        if (str(line).__contains__("b'PRINT ENTRY = ") and ok == 'true'):
            if (len(str(line).replace("b'ENTRY = ", "").strip("\n'").split("|")) > 1):
                seed_word = str(line).replace("b'PRINT ENTRY = ", "").strip("\n'").split("|")[0]
                count += 1
                if seed_words != "":
                    #dictionary[heading] = dictionary[heading] + "#" + seed_word
                    seed_words=seed_words+"#"+seed_word
                elif seed_words == "":
                    #dictionary[heading] = seed_word
                    seed_words=seed_word
        if (str(line).__contains__("b'ENTRY = ") and ok=='true'):
            if(len(str(line).replace("b'ENTRY = ", "").strip("\n'").split("|")) > 1):
                seed_word = str(line).replace("b'ENTRY = ", "").strip("\n'").split("|")[0]
                count+=1
                if seed_words != "":
                    #dictionary[heading] = dictionary[heading] + "#" + seed_word
                    seed_words = seed_words + "#" + seed_word
                elif seed_words == "":
                    #dictionary[heading] = seed_word
                    seed_words = seed_word
        if(str(line).__contains__("b'MN = ") and ok=='true'):
            tree.add(str(line).replace("b'MN = ","").strip("\n'")[0])
        if(str(line).__contains__("b'PI = ") and ok=='true'):
            new_line= str(line)[:str(line).rfind("(")]
            previous_indexing= new_line.replace("b'PI = ","")
            if previous_indexings != "":
                previous_indexings = previous_indexings + "#" + previous_indexing
            elif previous_indexings == "":
                previous_indexings=previous_indexing
        if (str(line).__contains__("b'MS = ") and ok == 'true'):
            description= str(line).replace("b'MS = ", "").strip("\n'")[:-2]
            #dictionary[heading]= [seed_words,count,description,tree,previous_indexings]
            #ok='false'
        if (str(line).__contains__("b'PM = ") and ok == 'true'):
            #print(line)
            public_mesh_note= str(line).replace("b'PM = ", "").strip("\n'")[:-2]
            if(len(public_mesh_note) <=11):
                public_mesh_note=""
            #dictionary[heading]= [seed_words,count,description,tree,previous_indexings,public_mesh_note]
            #ok='false'
        if(str(line).__contains__("b'UI = ") and ok == 'true'):
            dictionary[heading]= [seed_words,count,description,tree,previous_indexings,public_mesh_note]
            ok='false'
    return dictionary


dicti=create_seed_word_list("d2020.bin")

"""
found=0
for key in dicti.keys():
    #if(dicti[key][5]!='' and not (dicti[key][5].lower().__contains__('see') or dicti[key][5].lower().__contains__('under') or dicti[key][5].lower().__contains__('was') or dicti[key][5].lower().__contains__("use") or dicti[key][5].lower().__contains__('were'))):
        print(key)
        print("#################")
        print(dicti[key])
        #print(dicti[key][5].split("under")[1])
        #print(len(dicti[key][5]))
        found+=1

print(len(dicti))
print(found)
"""



#%%
def read_labels(f):
    file = open(f)
    top_labels = list()
    for line in file:
        top_labels.append(line[:-1])

    return top_labels



def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

################# Extracting the Descriptor information from the Public Mesh Note Field for the top 100 labels##########
count=0
for key in dicti.keys():
        previousupper = False
        previous = False
        final_item = ""
        #print(key)
        count+=1
        #print(dicti[key][5])
        public_mesh_note=dicti[key][5].split(" ")
        if(dicti[key][5]!=''):
            dicti[key][5]=list()
        for item in public_mesh_note:
            if(item.isupper() and previousupper==False):
                previousupper=True
                final_item=item
                #print(item)
            elif(item.isupper() and previousupper == True):
                final_item=final_item+" "+item
            elif(previousupper == True and not item.isupper()):
                #print(final_item)
                if(final_item.lower() != key.lower() and final_item not in dicti[key][5]):
                    dicti[key][5].append(final_item)
                previousupper=False
            elif(item.__contains__('was') or item.__contains__('use') or item.__contains__('see')):
                previous = True
            elif(previous== True and not hasNumbers(item)):
                if(final_item== ""):
                    final_item=item
                else:
                    final_item=final_item+" "+item
            elif(previous == True and final_item!=""):
                previous= False
                #print(final_item)

        if(final_item not in dicti[key][5]):
            dicti[key][5].append(final_item)


count=0
new_count=0
complex_count=0
f = open('completely_new_labels.txt', "a")
f2 = open('new_labels_from_complex_changes.txt', "a")
f3 = open("previous_host_labels.txt","a")
previous_hosts=set()
"""
for key in dicti.keys():
    if key in top_labels:
        if(dicti[key][5] == ''):
            new_count+=1
            f.write(str(key))
            f.write("\n")    
        elif(dicti[key][5] != ''):
            complex_count+=1
            f2.write(str(key))
            f2.write("\n")
"""
not_count=0
not_ph=list()
for key in dicti.keys():
        not_count=0
        if(dicti[key][4] ==''):
            for item in dicti[key][4].split("#"):
                previous_hosts.add(item)
                not_count+=1
            print(":D")
        if (dicti[key][5] == ''):
            for item in dicti[key][5]:
                previous_hosts.add(item)
            not_count+=1
            print(":DD")
        if(not_count == 1):
            print(dicti[key][4])
            print(dicti[key][5])
        if(not_count == 2):
            not_ph.append(key)

print(len(dicti.keys()))
print("Labels without previous host: "+str(len(not_ph)))


for item in previous_hosts:
    f3.write(str(item))
    f3.write("\n")
print(count)
print(new_count)
print(complex_count)


with open("label_info.pickle","wb") as f:
    pickle.dump(dicti, f)
    f.close()









