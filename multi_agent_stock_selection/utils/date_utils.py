from datetime import datetime
from dateutil.relativedelta import relativedelta


def get_begin_date(start_date: int | str, year: int = 1, month: int = 2, day: int = 0) -> int:
    input_date = datetime.strptime(str(start_date), "%Y%m%d")
    begin_date = input_date - relativedelta(years=year, months=month, days=day)
    begin_date_str = begin_date.strftime("%Y%m%d")
    return int(begin_date_str)
