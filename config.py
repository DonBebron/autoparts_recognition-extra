import time


class Config():

  mainpage_url = "https://auctions.yahoo.co.jp/category/list/2084017107/?p=アウディ用&auccat=2084017107&istatus=2%2C1&is_postage_mode=0&dest_pref_code=13&exflg=1&b=1&n=100&s1=new&o1=d&brand_id=118482"
  model_path = 'checkpoint.weights.h5'

  image_size = (512, 512)
  image_channels = 3
  image_shape = (*image_size, image_channels)

  batch_size = 32

class Logs():
  runtimes = ''

  def __call__(self, log_text):
    print(log_text)
    self.runtimes += f"\n{log_text}"
    return self.runtimes

  def pop(self):

    runtimes_text = str(self.runtimes)
    self.runtimes = ''
    return runtimes_text


class RuntimeMeta(type):
    def __new__(cls, name, bases, dct):
        for attr, value in dct.items():
            if callable(value):
                dct[attr] = cls.wrap_with_runtime(value)
        return super(RuntimeMeta, cls).__new__(cls, name, bases, dct)

    @staticmethod
    def wrap_with_runtime(func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            print(f"Runtime of {func.__name__}: {end_time - start_time:.4f} seconds")
            return result
        return wrapper

