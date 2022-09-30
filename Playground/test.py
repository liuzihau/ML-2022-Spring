import numpy as np
import datetime
datetime_today_obj = datetime.datetime.today()
today_string = datetime_today_obj.strftime('%Y_%m_%d_(%H:%M:%S)')
print(np.arange(50))
print(today_string)

a = np.array([[[1,2,3,4],[2,2,3,4],[3,2,4,5]],[[1,2,3,4],[2,2,3,4],[3,2,4,5]],[[1,2,3,4],[2,2,3,4],[3,2,4,5]]])
print(a.shape[2:])