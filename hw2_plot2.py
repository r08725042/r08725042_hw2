
# coding: utf-8

# In[4]:


data=[]
temp = {'name':'0.1', 'overall_em':0.83492252681764, 'overall_f1':0.8652080689398022, 'answerable_em':0.7778093076049943,
       'answerable_f1':0.8210719123277422, 'unanswerable':0.9682119205298013}
data.append(temp)
temp = {'name':'0.3', 'overall_em':0.83492252681764, 'overall_f1':0.8652080689398022, 'answerable_em':0.7778093076049943,
       'answerable_f1':0.8210719123277422, 'unanswerable':0.9682119205298013}
data.append(temp)

temp = {'name':'0.5', 'overall_em':0.8359157727453318, 'overall_f1':0.8662013148674939, 'answerable_em':0.7775255391600454,
       'answerable_f1':0.8207881438827933, 'unanswerable':0.9721854304635762}
data.append(temp)

temp = {'name':'0.7', 'overall_em':0.8351211760031784, 'overall_f1':0.86500800083151, 'answerable_em':0.7761066969353008,
      'answerable_f1':0.8187997378506868, 'unanswerable':0.9728476821192052}
data.append(temp)

temp = {'name':'0.9', 'overall_em':0.834326579261025, 'overall_f1':0.8638455352272485, 'answerable_em':0.7735527809307605,
       'answerable_f1':0.8157203247258707, 'unanswerable':0.976158940397351}
data.append(temp)


# In[54]:


overall_em = [i['overall_em']for i in data]
answerable_em = [i['answerable_em'] for i in data]
unanswerable = [i['unanswerable'] for i in data]
overall_f1 = [i['overall_f1'] for i in data]
answerable_f1 = [i['answerable_f1'] for i in data]


# In[138]:


import matplotlib.pyplot as plt
plt.figure()
plt.figsize=(8, 16)

plt.dpi = 500
x = [0.1,0.3,0.5,0.7,0.9]
plt.suptitle('Performance on different threshold')
plt.subplot(1,2,1)
plt.plot(x, overall_em,'o-',color = 'b', label='overall_em')
plt.plot(x, answerable_em,'o-',color = 'tab:orange', label='answerable_em')
plt.plot(x, unanswerable,'o-',color = 'g', label='unanswerable')
plt.xticks(x,('0.1','0.3','0.5','0.7','0.9'))
plt.xlabel('Answerable Threshold')


plt.subplot(1,2,2)
plt.plot(x, overall_f1,'o-',color = 'b', label='overall')
plt.plot(x, answerable_f1,'o-',color = 'tab:orange', label='answerable')
plt.plot(x, unanswerable,'o-',color = 'g', label='unanswerable')
plt.xticks(x,('0.1','0.3','0.5','0.7','0.9'))
plt.legend(loc = (0.7,0.7), fontsize=8)
plt.subplots_adjust(wspace =0.3)
plt.xlabel('Answerable Threshold')



# In[139]:


plt.savefig('a.png')

