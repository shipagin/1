from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#база данных ирисов
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
#имена столбцов
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#чтение базы данных по столбцам указанным в names
setofdata = read_csv(url, names=names)

# --------проверка данных----------
# количество элементов и классов
#print(setofdata.shape)

# Стастическая сводка
#print(setofdata.describe())

# Распределение по атрибуту class
#print(setofdata.groupby('class').size())

#вид графика box, графики отдельно дрг от друга, размер окна 2*2
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=True, sharey=False)
#pyplot.show()

# Гистограмма распределения атрибутов датасета
#setofdata.hist()
#pyplot.show()

#Матрица диаграмм рассеяния
#scatter_matrix(setofdata)
#pyplot.show()

#----------создание выборки данных--------
# Берем значения из представленной базы
array = setofdata.values

# Выбор первых 4-х столбцов 
Input = array[:,0:4]

# Выбор 5-го столбца 
Target = array[:,4]

# Разделение X и y на обучающую и валидационную выборки 
Input_tr, Input_val, Target_tr, Target_val = train_test_split(Input, Target, test_size=0.2, random_state=1)


#------------тестирование различных моделей машинного обучения---------
# Загружаем алгоритмы модели
methods = []
methods.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
methods.append(('LDA', LinearDiscriminantAnalysis()))
methods.append(('KNN', KNeighborsClassifier()))
methods.append(('CART', DecisionTreeClassifier()))
methods.append(('NB', GaussianNB()))
methods.append(('SVM', SVC(gamma='auto')))

# оцениваем модель на каждой итерации
results = []
names = []

for name, method in methods:
        # используем стратифицированную 10-кратную кросс-валидацию с перемешиванием
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        # тестирование методов по выборке
	m_results = cross_val_score(method, Input_tr, Target_tr, cv=kfold, scoring='accuracy')
        # добавление результатов в список
	results.append(m_results)
        # добавление имени модели
	names.append(name)
        # на экран выводим результат по обучению для каждого алгоритма + стандартное отклонение
        # по точности и отклонению нас больше всего устраивает SVC модель
	print('%s: %s %f %s %f' % (name, 'accuracy:', m_results.mean(), 'std:', m_results.std()))

# Сравниванием алгоритмы по графикам (SVC имеет наибольшую точность и наименьший разброс значений в зависимости от среза)
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()


#-------------проверка выбранной модели по данным на валидационной выборке (модель SVM)-------------

# Формирование значений по валидлационной выборке
# Выбор SVC модели
method = SVC(gamma='auto')
# обучение модели на учебной выборке
method.fit(Input_tr, Target_tr)
#прогнозирование значений для валидационных данных
predictions = method.predict(Input_val)

#Оценка полученных результатов по валидационной выборке
# оценка точности по валидационной выборке
print('accuracy of SVM method on Input_val: ', accuracy_score(Target_val, predictions))
#построение матрицы ошибок (как видно из нее - была допущена всего 1 ошибка)
print ('Confussion matrix:')
print( confusion_matrix(Target_val, predictions))
# просмотр каждого класса по точности
print('report for every class: ')
print(classification_report(Target_val, predictions))
