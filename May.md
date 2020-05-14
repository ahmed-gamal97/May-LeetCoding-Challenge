# May LeetCoding Challenge

1) https://leetcode.com/problems/first-bad-version </br>
You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad. Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad. You are given an API bool isBadVersion(version) which will return whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API

```python
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        l = 1
        r = n
        
        while l <= r:
            
            mid = (l+r) // 2
            
            if isBadVersion(mid):
                if not isBadVersion(mid-1):
                    return mid
                else:
                    r = mid - 1
            else:
                l = mid + 1
        
```
### Complexity: O(log(n)) , space: O(1)
----------------------
2) https://leetcode.com/problems/jewels-and-stones/ </br>
You're given strings J representing the types of stones that are jewels, and S representing the stones you have. Each character in S is a type of stone you have.  You want to know how many of the stones you have are also jewels. The letters in J are guaranteed distinct, and all characters in J and S are letters. Letters are case sensitive, so "a" is considered a different type of stone from "A".

```python
class Solution:
    def numJewelsInStones(self, J: str, S: str) -> int:
        
        jewels = {}
        
        for letter in J:
            jewels[letter] = 1
             
        # Stones you have are also jewels.
        stones_jewels_couunter = 0
        
        for letter in S:
            if letter in jewels:
                stones_jewels_couunter += 1
            
        return stones_jewels_couunter
        
```
### Complexity: O(n) , space: O(n)
-----------------------

3) https://leetcode.com/problems/ransom-note/ </br>
Given an arbitrary ransom note string and another string containing letters from all the magazines, write a function that will return true if the ransom note can be constructed from the magazines ; otherwise, it will return false.Each letter in the magazine string can only be used once in your ransom note.

```python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        
        ransom_dict = {}
        
        for letter in ransomNote:
            if letter in ransom_dict:
                ransom_dict[letter] += 1
            else:
                ransom_dict[letter] = 1
                
                
        for letter in magazine:
            if letter in ransom_dict:
                ransom_dict[letter] -= 1
                if ransom_dict[letter] == 0:
                    del ransom_dict[letter]
            if len(ransom_dict) == 0:
                return True
                    
        return len(ransom_dict) == 0
        
```
### Complexity: O(n) , space: O(n)
-----------------------
4) https://leetcode.com/problems/number-complement/ </br>
Given a positive integer, output its complement number. The complement strategy is to flip the bits of its binary representation.

```python
class Solution:
    def findComplement(self, num: int) -> int:
        
        number_in_binary = bin(num)[2:]
        
        number_in_binary = ['1' if char == '0' else '0' for char in number_in_binary]
        
        return int(''.join(number_in_binary) , 2)

```
### Complexity: O(num of digits) , space: O(num of digits)
-----------------------
5) https://leetcode.com/problems/first-unique-character-in-a-string/ </br>
Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        
        char_counter = {}
        
        for char in s:
            if char in char_counter:
                char_counter[char] += 1
            else:
                char_counter[char] = 1
                
        for ind,char in enumerate(s):
            if char_counter[char] == 1:
                return ind
        
        return -1
```
### Complexity: O(len(s)) , space: O(len(s))
-----------------------
6) https://leetcode.com/problems/majority-element/ </br>
Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times. You may assume that the array is non-empty and the majority element always exist in the array.
```python
import collections

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        
        num_freq = collections.Counter(nums)
        
        length = len(nums)
        half = length // 2
        
        for num in nums:
            if num_freq[num] > half:
                return num
```
### Complexity: O(len(nums)) , space: O(len(nums))
-----------------------
7) https://leetcode.com/problems/cousins-in-binary-tree/ </br>
- In a binary tree, the root node is at depth 0, and children of each depth k node are at depth k+1.
- Two nodes of a binary tree are cousins if they have the same depth, but have different parents.
- We are given the root of a binary tree with unique values, and the values x and y of two different nodes in the tree.
- Return true if and only if the nodes corresponding to the values x and y are cousins.
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        
        if root.val in [x,y]:
            return False
        
        depth = 0
        parent = None
        queue = [(root,parent,depth)]

        nodes_info = []
        
        # BFS
        while queue:
            
            node,_,depth = queue.pop(0)
            
            if node.left:
                queue.append((node.left, node, depth+1)) 
                if node.left.val in [x,y]:
                    nodes_info.append((node.val, depth+1))
                    
            if node.right:
                queue.append((node.right, node, depth+1))
                if node.right.val in [x,y]:
                    nodes_info.append((node.val, depth+1))

            if len(nodes_info) == 2:
                break
                
        return nodes_info[0][1] == nodes_info[1][1] and nodes_info[0][0] != nodes_info[1][0]
```
### Complexity: O(#nodes) , space: O(#nodes)
-----------------------
8) https://leetcode.com/problems/check-if-it-is-a-straight-line/ </br>
You are given an array coordinates, coordinates[i] = [x, y], where [x, y] represents the coordinate of a point. Check if these points make a straight line in the XY plane.
```python
class Solution(object):
    def checkStraightLine(self, coordinates):
        """
        :type coordinates: List[List[int]]
        :rtype: bool
        """
        try:
            slope = (coordinates[1][1] - coordinates[0][1]) / (coordinates[1][0] - coordinates[0][0])
        except ZeroDivisionError:
            return False
        
        for ind in range(len(coordinates)-1, 0, -1):
            try:
                to_cal = (coordinates[ind][1] - coordinates[ind-1][1]) / (coordinates[ind][0] - coordinates[ind-1][0])
                if slope != to_cal:
                    return False
            
            except ZeroDivisionError:
                return False
            
        return True
```
### Complexity: O(n) , space: O(1)
-----------------------
9) https://leetcode.com/problems/valid-perfect-square/ </br>
- Given a positive integer num, write a function which returns True if num is a perfect square else False.
- Note: Do not use any built-in library function such as sqrt.
```python
class Solution(object):
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        start = 0
        end = num      
        
        while start <= end:
            
            mid = (start + end) // 2
            test = mid ** 2
            
            if test == num:
                return True
            elif test > num:
                end = mid - 1
            else:
                start = mid + 1
                
        return False
    
        # Math Trick for Square number is 1+3+5+ ... +(2n-1)
        def Math(self, num):
            i = 1
            while (num>0):
                num -= i
                i += 2       
            return num == 0
```
### Complexity: O(log(n)) , space: O(1)
-----------------------
10) https://leetcode.com/problems/find-the-town-judge/ </br>
- In a town, there are N people labelled from 1 to N.
- There is a rumor that one of these people is secretly the town judge 
If the town judge exists, then the town judge trusts nobody.
- Everybody (except for the town judge) trusts the town judge.
- There is exactly one person that satisfies properties 1 and 2.
- You are given trust, an array of pairs trust[i] = [a, b] representing that the person labelled a trusts the person labelled b.

If the town judge exists and can be identified, return the label of the town judge.  Otherwise, return -1.
```python
class Solution:
    def findJudge(self, N: int, trust: List[List[int]]) -> int:
        
        if N == 1 and not trust:
            return 1
        
        unique_keys = set()
        val_counter = {}
        
        for a,b in trust:
            
            unique_keys.add(a)
            
            if b in val_counter:
                val_counter[b] += 1
            else:
                val_counter[b] = 1
        
        for i in range(1, N+1):
            if i not in unique_keys and i in val_counter and val_counter[i] == N-1:
                return i
            
        return -1
```
### Complexity: O(n) , space: O(n)
-----------------------
11) https://leetcode.com/problems/flood-fill/ </br>
- An image is represented by a 2-D array of integers, each integer representing the pixel value of the image (from 0 to 65535).
- Given a coordinate (sr, sc) representing the starting pixel (row and column) of the flood fill, and a pixel value newColor, "flood fill" the image.
- To perform a "flood fill", consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color as the starting pixel), and so on. Replace the color of all of the aforementioned pixels with the newColor.
- At the end, return the modified image.
If the town judge exists and can be identified, return the label of the town judge.  Otherwise, return -1.
```python
class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        
        # BFS
        
        MIN_ROW = 0
        MIN_COL = 0
        
        MAX_ROW = len(image)
        MAX_COL = len(image[0])
        
        STARTING_PIXEL = image[sr][sc]
        
        queue = [(sr,sc)]
        mem = {(sr,sc) : 0}
        
        while len(queue) != 0 :
            
            r,c = queue.pop(0)
            
            if r-1 >= MIN_ROW and image[r-1][c] == STARTING_PIXEL and (r-1,c) not in mem:
                image[r-1][c] = newColor
                queue.append((r-1,c))
                mem[(r-1,c)] = 1
                
            if r+1 < MAX_ROW and image[r+1][c] == STARTING_PIXEL and (r+1,c) not in mem:
                image[r+1][c] = newColor
                queue.append((r+1,c))
                mem[(r+1,c)] = 1
                
            if c-1 >= MIN_COL and image[r][c-1] == STARTING_PIXEL and (r,c-1) not in mem:
                image[r][c-1] = newColor
                queue.append((r,c-1))
                mem[(r,c-1)] = 1

            if c+1 < MAX_COL and image[r][c+1] == STARTING_PIXEL and (r,c+1) not in mem:
                image[r][c+1] = newColor
                queue.append((r,c+1))
                mem[(r,c+1)] = 1
            
        image[sr][sc] = newColor

        return image
```
### Complexity: O(n*m) , space: O(n*m)
-----------------------
12) https://leetcode.com/problems/single-element-in-a-sorted-array/ </br>
- You are given a sorted array consisting of only integers where every element appears exactly twice, except for one element which appears exactly once. Find this single element that appears only once.
- Note: Your solution should run in O(log n) time and O(1) space.
```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        
        length = len(nums)
        
        if length == 1:
            return nums[0]
        
        l = 0
        r = length - 1
        
        while l < r:
            
            mid = (r + l) // 2
            
            if mid & 1: # is odd
                if nums[mid] == nums[mid-1]:
                    l = mid + 1
                else:
                    r = mid - 1
                    
            else: # is even
                if nums[mid] == nums[mid+1]:
                    l = mid + 2
                else:
                    r = mid
                    
        return nums[r]
```
### Complexity: O(log n) , space: O(1)
-----------------------
13) https://leetcode.com/problems/remove-k-digits/ </br>
- Given a non-negative integer num represented as a string, remove k digits from the number so that the new number is the smallest possible. 
- Note:
- The length of num is less than 10002 and will be ≥ k.
- The given num does not contain any leading zero.

```python
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
       
        length = len(num)
        
        if k == length:
            return '0'
        
        stack = []
        
        for i in range(length):
        # remove the tail of stack if the element is smaller than the stack tail
            while k and stack and num[i] < stack[-1]:
                stack.pop(-1)
                k -= 1
            stack.append(num[i])

        # If still k elements left to remove.
        while k > 0 and stack:
            stack.pop(-1)
            k -= 1
            
        # Remove leading zeros
        res = ''.join(stack).lstrip("0")
        
        return res if res else '0'
        
```
### Complexity: O(n) , space: O(n)
-----------------------
14) https://leetcode.com/problems/implement-trie-prefix-tree/
 </br>
- Implement a trie with insert, search, and startsWith methods.
- You may assume that all inputs are consist of lowercase letters a-z.
- All inputs are guaranteed to be non-empty strings.

```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = {}

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self
        for char in word:
            if char not in cur.children:
                cur.children[char] = Trie()
            cur = cur.children[char]
        cur.children['.'] = '/0' 

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        curr = self
        if len(word) == 0:
            return True
        for char in word:
            if char in curr.children:
                curr = curr.children[char]
            else:
                return False
        return '.' in curr.children

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        curr = self
        if len(prefix) == 0:
            return True
        for char in prefix:
            if char in curr.children:
                curr = curr.children[char]
            else:
                return False
        return True

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```
### Complexity: O(n) , space: O(n)
-----------------------