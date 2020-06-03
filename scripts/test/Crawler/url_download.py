from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"tiger",
             "limit":10000,"print_urls":True,
             "chromedriver":"C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images