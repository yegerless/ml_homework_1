# Домашнее задание №1 (базовая группа)
##### Применение линейной регрессии для предсказания стоимости автомобилей.
Задание выполнил студент магистратуры ВШЭ "Искусственный интеллект" Сысоев Сергей Александрович.


## EDA и обучение моделей
### Характеристика данных

Набор данных состоял из 7999 строк (6999 в тренировочном датасете и 1000 в тестовом датасете) и 13 столбцов:

- name - название модели автомобиля
- year - год выпуска
- selling_price - стоимость (целевая переменная)
- km_driven - пробег
- fuel - тип топлива
- seller_type - тип продавца
- transmission - тип коробки передач
- owner - порядковый номер текущего владельца
- mileage - расход топлива?
- engine - объем двигателя
- max_power - можность двигателя
- torque - крутящий момент двигателя
- seats - количество посадочных мест

#### Пропуски в данных
Обнаруженные пропущенные значения в тренировочном датасете:
- В столбце "mileage" находится 202 пропущенных значений 
- В столбце "engine" находится 202 пропущенных значений 
- В столбце "max_power" находится 196 пропущенных значений 
- В столбце "torque" находится 203 пропущенных значений В столбце "seats" находится 202 пропущенных значений

Обнаруженные пропущенные значения в тестовом датасете:
- В столбце "mileage" находится 19 пропущенных значений 
- В столбце "engine" находится 19 пропущенных значений 
- В столбце "max_power" находится 19 пропущенных значений 
- В столбце "torque" находится 19 пропущенных значений 
- В столбце "seats" находится 19 пропущенных значений

Столбец torque был удален. Из значений в столбцах mileage, engine и max_power были удалены единицы измерения, затем значения были конвертированы в тип данных float. Все пропущенные значения были заменены медианами соответствующего признака. Данное преобразование было выполнено как в тренировочном, так и в тестово датасете.

#### Дублирующиеся значения
В тренировочном датасете обнаружено 985 дублирующихся строчек, а в тестовом - 62 дублирующие строчки.  
Без учета целевой переменной в тренировочном датасете обнаружено 1159 дублирующихся строк. Дублирующие строки в тренировочном датасете были удалены.




### EDA
#### Описательные статистики
Числовые данные:
В обоих датасетах (тренировочный и тестовый) были автомобили в среднем 2013 года выпуска, средний mileage составил около 19, среднее количество посадочных мест составило 5. При этом в тестовом датасете были выше средняя цена автомобиля (617901 против 535432 в тренировочном), объем двигателя (1454 против 1429 в тренировочном датасете), и максимальная мощность (примерно 91 против примерно 88). В тренировочном датасете оказался выше средний пробег автомобиля (73952 против 71393 в тестовом).

Категориальные данные:
В тренировочном датасете было 1924 уникальных моделей автомобилей (и наиболее часто встречающаяся Maruti Alto 800 LXI), когда в тестовом только 621 (и наиболее часто встречающаяся Maruti Alto 800 LXI). Признаки fuel, seller_type, transmission, owner, были схожи образом в обоих датасетах: в fuel 4 категории и наиболее часто встречающаяся Diesel, в seller_type 3 категории и Individual, в transmission 2 категории и Manual, в owner 5 категорий и First Owner.

Совокупности распределения признаков в тренировочном и тестовом датасетах схожи.

#### Связь между признаками
Признаки engine и max_power, положительно коррелируют между собой (коэффициент корреляции 0.68), что логично, так как за увеличением объема двигателя обычно следует увеличение его мощности. Также положительно коррелируют признаки engine и seats (коэффициент корреляции 0.65).
Признак max_power положительно коррелирует с целевой переменной (коэффициент корреляции 0.69).

#### Дополнительные иснайды
- Автоматический тип трансмиссии характерен для более дорогих автомобилей
- Чем меньше владельцев было у автомобиля, тем выше медианная стоимость. Самую высокую стоимость демонстрируют автомобили из категории Test Drive Car.





### Обучение моделей
В ходе выполнения задания было построено несколько моделей линейной регрессии:
1) Линейная регрессия на числовых признаках
2) Линейная регрессия на масштабированных числовых признаках
3) Lasso регрессия на масштабированных числовых признаках
4) Lasso регрессия на масштабированных числовых признаках и подобранном гиперпараметре alpha кросс-валидацией на 10 фолдах при помощи GridSearchCV
5) ElasticNet регрессиия на масштабированных числовых признаках и подобранными гиперпараметрами alpha и l1-ratio кросс-валидацией на 10 фолдах при помощи GridSearchCV
6) Ridge регрессия на числовых и категориальных (закодированы при помощи OHE) признаках и подобранном гиперпараметре alpha кросс-валидацией на 10 фолдах при помощи GridSearchCV
7) Ridge регрессия на масштабированных числовых и категориальных (закодированы при помощи OHE) признаках и подобранном гиперпараметре alpha кросс-валидацией на 10 фолдах при помощи GridSearchCV

##### Линейная регрессия на числовых признаках
В качестве базовой самой простой модели была обучена классическая линейная регрессия (LinearRegression) на датасете, который состоял только из числовых признаков. В результате MSE составил 233297548204.61237, а коэффициент r2 составил 0.594. Качество этой модели оказалось низким, она плохо предсказывает целевую переменную (высокий MSE, низкий r2).


##### Линейная регрессия на масштабированных числовых признаках
Для улучшения качества базовой модели было выполнено масштабирование признаков при помощи StandardScaler, обученного на тестовой выборке. Однако это не дало прирост качества модели, значения MSE и r2 не изменились (так как масштабирование признаков не влияет на точность классической линейной регрессии). Тем не менее, масштабирование признаков открыло возможность для интерпретации важности признаков модели. Наиболее высоким оказался вес признака max_power, а значит он наиболее важен для прогнозирования целевой переменной.


##### Lasso регрессия на масштабированных числовых признаках
Затем на все тех же масштабированных данных была обучена Lasso регрессия с параметрами по-умолчанию. Данная модель не улучшила точность прогнозирования целевой переменной, метрики MSE и коэффициент r2 изменились незначительно (MSE вырос на 60000, r2 понизился на 1 десятимиллионную).


##### Lasso регрессия на масштабированных числовых признаках и подобранном гиперпараметре alpha кросс-валидацией на 10 фолдах при помощи GridSearchCV
Для повышения качества предыдущей модели необходимо подобрать оптимальные значения гиперпараметра alpha. Для этого Lasso регрессия была обернута в GridSearchCV, в которой гиперпараметр подбирался при помощи кросс-валидации на 10 фолдах. В результате подобрано оптимальное значение alpha 10000. Затем для лучшей модели из GridSearchCV были рассчитаны метрики качества на тестовой выборке, MSE составил 240512206991.89523, r2 составил 0.582. Таким образом эта модель оказалась хуже классической линейной регрессии, так как значение MSE выросло, а значение коэффициента r2 понизилось.


##### ElasticNet регрессиия на масштабированных числовых признаках и подобранными гиперпараметрами alpha и l1-ratio кросс-валидацией на 10 фолдах при помощи GridSearchCV
Далее была обучена ElasticNet регрессия, гиперпараметры (alpha, l1-ratio) к которой были также подобраны на 10 фолдах при помощи GridSearchCV. Оптимальные значения были подобраны следующие: alpha=1, l1_ratio=0.9. Затем для лучшей модели из GridSearchCV были рассчитаны метрики качества на тестовой выборке, MSE составил 245892672007.79794, коэффициент r2 составил 0.572. Таким образом, эта модель оказалась хуже предыдущей и хуже классической линейной регрессии, так как значение MSE выросло, а значение коэффициента r2 понизилось.


##### Ridge регрессия на числовых и категориальных (закодированы при помощи OHE) признаках и подобранном гиперпараметре alpha кросс-валидацией на 10 фолдах при помощи GridSearchCV
Для улучшения качества модели к числовым данным были добавлены категориальны, которые предварительно были закодированы при помощи OneHotEncoder. Гиперпараметр alpha был подобран кросс-валидацией на 10 фолдах при помощи GridSearchCV, оптимальное значение составило 10. Затем для лучшей модели из GridSearchCV были рассчитаны метрики качества на тестовой выборке, MSE составил 207550670033.712, коэффициент r2 составил 0.639. Таким образом, эта модель оказалась лучше всех предыдущих, на ней MSE снизился, а коэффициент r2 наоборот вырос. Но стоит отметить, что само по себе качество этой модели остается достаточно низким.


##### Ridge регрессия на масштабированных числовых и категориальных (закодированы при помощи OHE) признаках и подобранном гиперпараметре alpha кросс-валидацией на 10 фолдах при помощи GridSearchCV
Для улучшения качества предыдущей модели все данные были масштабированы при помощи StandardScaler. Затем был подобран гиперпараметр alpha с помощью кросс-валидации на 10 фолдах (GridSearchCV), оптимальное значение alpha составило 1000. Затем для лучшей модели из GridSearchCV были рассчитаны метрики качества на тестовой выборке, MSE составил 230961738812.60727, коэффициент r2 составил 0.598. После масштабирования признаков качество модели ухудшилось, но она оказалась чуть лучше модели на классической линейной регрессии.


#### Оценка бизнес-метрики
Для дополнительной оценки качества всех обученных моделей использовалась следующая бизнес-метрика: доля прогнозов, отличающихся от реальных цен не более чем на 10%.
По оценке данной бизнес-метрики лучшей оказалась последняя модель (Ridge регрессия на всех масштабированных данных), со значением метрики 0.255, а лучшая модель по значениям метрик MSE и коэффициента r2 оказалась лишь на 4 месте со значением бизнес-метрики 0.233.



### Заключение
По расчетам метрик MSE и коэффициента r2 лучшей оказалась модель 6 (Ridge регрессия на числовых и категориальных (закодированы при помощи OHE) признаках и подобранном гиперпараметре alpha кросс-валидацией на 10 фолдах при помощи GridSearchCV), а по значению бизнес-метрики - модель 7 (Ridge регрессия на масштабированных числовых и категориальных (закодированы при помощи OHE) признаках и подобранном гиперпараметре alpha кросс-валидацией на 10 фолдах при помощи GridSearchCV).
Наибольший прирост в качестве дало использование всего набора признаков, в том числе закодированных OHE категориальных переменных.
Не получилось улучшить качество модели при помощи масштабирования данных и использования Lasso и ElasticNet регрессий.





## FastApi сервис с лучшей моделью
Реализован FastApi сервис с двумя входными точками:
- /predict_item - принимает один объект в формате JSON и возвращает предсказанное значение selling_price
    Пример работы: 
        ![Запрос и ответ сервиса](https://github.com/yegerless/ml_homework_1/blob/main/post_item.png)

- /predict_items - принимает csv файл с одним и более объектом и возвращает csv файл со всеми столбцами из переданного файла + добавленным столбцом с предсказанными значениями selling_price
    Пример работы:
        ![Переданный csv файл](https://github.com/yegerless/ml_homework_1/blob/main/test_csv.png)
        ![Запрос и ответ сервиса](https://github.com/yegerless/ml_homework_1/blob/main/post_items.png)
        ![Файл csv, который вернул сервис](https://github.com/yegerless/ml_homework_1/blob/main/response_csv.png)