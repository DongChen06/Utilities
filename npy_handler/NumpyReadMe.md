#### save a numpy array


### save to pkl
```
data = (1, 2, 3, [[1,2,3],[4,5,6]], ['a','b','c'])
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)
 ```
 read data
 ```
 with open('data.pickle', 'rb') as f:
     data = pickle.load(f)
 ```
