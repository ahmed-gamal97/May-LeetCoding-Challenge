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
### Complexity: O(log(n)) , space: o(1)
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
### Complexity: O(n) , space: o(n)
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
### Complexity: O(n) , space: o(n)
-----------------------