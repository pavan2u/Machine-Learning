import numpy as np


#Q1
def vowelOrConsonant():
    str=input("Enter a string: ")

    str1=str.lower()

    vowelcount=0
    consonantcount=0
    for i in str1:
      if (i == 'a' or i == 'e' or
           i == 'i' or i == 'o' or i == 'u'):
           vowelcount=vowelcount+1
      else:
           consonantcount=consonantcount+1
    
    return vowelcount,consonantcount
    
v, c = vowelOrConsonant()
print("The number of vowels: ",v)
print("The number of consonants: ", c)


#Q2
def input_matrix(rows, cols):
    print(f"Enter elements for a {rows}x{cols} matrix: ")
    elements = list(map(float, input().split()))
    return np.array(elements).reshape(rows, cols)

rows1 = int(input("Enter the number of rows for the first matrix: "))
cols1 = int(input("Enter the number of columns for the first matrix: "))
B = input_matrix(rows1, cols1)

rows2 = int(input("Enter the number of rows for the second matrix: "))
cols2 = int(input("Enter the number of columns for the second matrix: "))
A = input_matrix(rows2, cols2)

def matrixmultiplication():

    try:
        result= [[0,0,0,0],
                [0,0,0,0],
                [0,0,0,0]]
        result = np.dot(A,B)

        return result
    except:
        print("matrix dimensions are not multiplyable")


result=matrixmultiplication()


for r in result:
    print(r)



#Q3
def checkCommon():
    list1=list(input("Enter first list: "))
    list2=list(input("Enter second list: "))

    print(list1)
    print(list2)

    count=0
    for i in list1:
        for j in list2:
            if(i==j):
                count+=1
                break
    return count

common_elements=checkCommon()
print("The number of common elements are: ",common_elements)

#Q4
def input_matrix2(rows, cols):
    print(f"Enter elements for a {rows}x{cols} matrix: ")
    elements = list(map(float, input().split()))
    return np.array(elements).reshape(rows, cols)


rows = int(input("Enter the number of rows for the first matrix: "))
cols = int(input("Enter the number of columns for the first matrix: "))
original_matrix = input_matrix2(rows, cols)

transposed_matrix = original_matrix.transpose()
 
print("Original Matrix:")
print(original_matrix)
print("\nTransposed Matrix:")
print(transposed_matrix)