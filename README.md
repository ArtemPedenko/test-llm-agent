Создайте файл requirements.txt для отслеживания зависимостей:
pip freeze > requirements.txt
Это сохранит установленные пакеты (например, fastapi и uvicorn) в файл.

В будущем вы сможете установить их одной командой:
pip install -r requirements.txt

Запуск сервера
uvicorn main:app --reload
