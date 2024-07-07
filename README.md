# CBR DS Tech Task: CPI Forecasting + Flask

Проект решает следующие задачи:
* Автоматически скачивает наиболее актуальный файл с данными по ИПЦ с сайта Росстата (https://rosstat.gov.ru/statistics/price).
* Позволяет работать со скачанными вручную данными из Витрины данных ГКС (https://showdata.gks.ru/olap2/descr/report/277326).
* Данные Росстата преобразуются из исходного представления к предыдущему месяцу (MoM) к представлению к аналогичному месяцу прошлого года (YoY) с целью избавления от влияния сезонного фактора в прогнозируемом показателе. Данные ГКС также рассматриваются в формате YoY.
* Полученные данные используются для обучения ряда моделей и подбора гипперпараметров с помощью кросс-валидации, а также последующей оценки качества на основе RMSE и построения прогноза на 6 месяцев вперёд.
* Прогнозы на тестовой выборке строятся по методологии расширяющегося окна (подробнее - в документации).
* Результаты представляются в виде графиков на веб-сайте.

## Содердание репозитория
* `app.py`: веб-приложение на Flask.
  * `index` - базовая страница,
  * `/show` - страница для отображения графиков,
  * `/tune` - подбор гиперпараметров.
  
  В форме на начальной странице можно выбрать источник данных (актуальные данные с сайта Росстата или предварительно скачанные данные из Витрины данных ГКС). Для данных Росстата реализована возможность выбора начальной и конечной даты с помощью формы, а также выбора типа показателя (ИПЦ всех товаров и услуг, ИПЦ прод. товаров, ИПЦ непрод. товаров и ИПЦ услуг).
  
  Для того чтобы перейти на страницу с графиком по выбранному показателю, нужно нажать на кнопку `Show plot`. Чтобы затем вернуться на начальную страницу, нужно нажать на кнопку `Back to index page`. Для подбора гиперпараметров нужно на начальной странице нажать кнопку `Tune Hyperparameters`, после чего на странице будет отображаться таймер, отображающий прошедшее с запуска время.

* `templates`: папка с html-шаблонами для веб-приложения.
  * `index.html` - шаблоны, стили и функции для основной страницы,
  * `show.html` - шаблоны, стили и функции для страницы с графиками.

* `static`: папка с графиками в формате `.png`, генерируемыми с помощью matplotlib.
 В папке хранятся последние 10 построенных графиков, более ранние графики автоматически удаляются.
  
* `Dockerfile` и `docker-compose.yml`: Docker-файлы для создания образа.
  Создать контейнер на основе образа можно с помощью следующих команд:
    * `docker pull lysovandrey/cbr-ds-tech-task-flask:latest`
    * `docker run -d -p 5000:5000 --name my-container lysovandrey/cbr-ds-tech-task-flask:latest`

* `requirements.txt`: текстовый файл с зависимостями, необходимыми для работы проекта.

* `src`: основной модуль с кодом для обработки данных и построения моделей.
    * `exception.py` - код для кастомной обработки ошибок,
    * `logger.py` - код для записи логов в папку `logs` при выполнении скриптов,
    * `utils.py` - ряд функций, необходимых для работы скриптов проекта,
    * `components`: модуль с основными скриптами проекта.
      * `data_ingestion.py` - набор классов для загрузки данных ГКС и создания лаговых фичей.
        Число лагов `num_lags` можно изменить в этом файле.
      * `data_rosstat.py` - набор классов для загрузки данных Росстата и создания лаговых фичейю
        Число лагов `num_lags` можно изменить в этом файле.
      * `data_transformation.py` - набор классов для построения пайплайна предобработки данных.
        В текущей версии доступно только заполнение NA.
      * `model_training.py` - набор классов для определения используемых моделей, подбора гиперпараметров и обучения моделей на данных.
        Набор моделей и сетку гиперпараметрво можно изменитть в этом файле.
        Модели сохраняются как `.pkl` файлы, которые затем можно вызвать при построении графифков.
    * `pipeline`: модуль с заготовкой пайплайна для предсказаний.

      В текущей версии не используется.
* `artifacts`: папка с выгруженными данными и сохранёнными моделями.
    * `data_raw.xlsx` - выгруженный исходник с сайта Росстата,
    * `data.xlsx` - обработанные данные Росстата,
    * `data_filtered.xlsx` - промежуточный файл с данными Росстата, необходимый для выбора дат с помощью формы,
    * `data.csv` - файл с данными ГКС,
    * `train.csv` - файл, в котором сохраняется обучающий датафрейм.
      
      Источником могут быть как данные Росстата, так и ГКС.
      
      Объём тренировочной выборки фиксирован 80%.
    * `test.csv` - файл, в котором сохраняется тестовый датафрейм.
      
      Источником могут быть как данные Росстата, так и ГКС.
      
      Объём тестовой выборки фиксирован 20%.
    * `train_filtered.csv` - промежуточный файл с обучающими данными Росстата, необходим для выбора дат,
    * `test_filtered.csv` - промежуточный файл с тестовыми данными Росстата, необходим для выбора дат,
    * `model_{name}.pkl` - файлы с сохранёнными моделями.
* `manual_data_download`: папка для ручной проверки кода и хранилище для данных ГКС.
    * `data_process.ipynb` - sandbox-ноутбук для обработки данных ГКС.
    * `rosstat.ipynb` - sandbox-ноутбук для обработки данных Росстата.
    * `prediction.ipynb` - sandbox-ноутбук для построения графиков.
    * `data_raw.csv` - файл с данными ГКС (скачан вручную).

      При обновлении данных ГКС нужно загрузить обновлённый файл вместо старого вручную.
    * Другие `.xlsx` и `.csv` - проверочные файлы для работы ноутбуков.
## Интерфейс приложения
Начальная страница:

![Начальная страница](/images_readme/index.png)

График по данным Росстата:
![График по данным Росстата](/images_readme/show_plot_rosstat.png)

График по данным ГКС:
![График по данным ГКС](/images_readme/show_plot_gks.png)

Запуск подбора гиперпараметров:
![Запуск подбора гиперпараметров](/images_readme/tuning.png)
