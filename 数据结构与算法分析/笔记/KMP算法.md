### KMP算法

+ ***算法4.6***

```c
/*从pos处开始搜索模式串T在主串S中首次出现的位置，如果不存在，则返回0。
 * 如果查找成功，返回匹配的位置。*/
int Index_KMP(SString S, SString T, int pos, int next[]) {
	int i = pos;
	int j = 1;	  
	if(pos < 1) {
	return 0;
	}
	// 比较字符串
	while(i <= S[0] && j <= T[0]) {
		/*
		* 两种情形：
		* 1.在模式串的第一个字符处就失配
		* 2.主串和模式串处的字符相等
		*/
		if(j == 0 || S[i] == T[j]) {
			i++;	
			j++;
   		} else {
		// 失配时回到前一个适当的位置
			j = next[j];
		}
	}
	if(j > T[0]) {
        // 匹配成功，返回匹配位置
        return i - T[0];
    } else {
        // 匹配失败
        return 0;
    }
}    
```
+ ***算法4.7***
  <img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20211201214906535.png" alt="image-20211201214906535" style="zoom: 50%;" />
  
  ```c
  /*
  *计算模式串的“失配数组”，用于KMP算法。
  */
  void get_next(SString T, int next[]) {
  	int i = 1;
  	int j = 0;
  	// 模式串第一个字符处失配时，模式串需要从头比较，主串需要前进到下一个位置比较
  	next[1] = 0;
  	// 遍历模式串上的字符
  	while(i < T[0]) {
    		if(j == 0 || T[i] == T[j]) {
        		i++;
        		j++;
        		next[i] = j;
    		} else {
        		j = next[j];
    		}
  	}
  }
  ```
  
+ 算法4.8

***next[j] = k,而模式中p_j= p_k，则当主串中字符s_i与p_j比较不等时，不与p_k进行比较，而是与p_next[k]进行比较***

```C
/*
 * 计算模式串的“失配数组”，用于KMP算法。
 * 这是一个优化后的版本，效率较算法4.7有所提高。
 */
void get_nextval(SString T, int nextval[]) {
    int i = 1;
    int j = 0;
    
    // 模式串第一个字符处失配时，模式串需要从头比较，主串需要前进到下一个位置比较
    nextval[1] = 0;
    
    // 遍历模式串上的字符
    while(i < T[0]) {
        if(j==0 || T[i] == T[j]) {
            i++;
            j++;
            
            if(T[i] != T[j]) {
                nextval[i] = j;
            } else {
                nextval[i] = nextval[j];
            }
        } else {
            j = nextval[j];
        }
    }
}
```
