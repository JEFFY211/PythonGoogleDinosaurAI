from selenium import webdriver
from selenium.webdriver.common.keys import Keys

profile = webdriver.FirefoxProfile()
profile.set_preference('webdriver.load.strategy', 'unstable')
try:
	browser = webdriver.Firefox(profile)
	browser.set_page_load_timeout(1)
	browser.set_script_timeout(1)
	browser.set_window_position(1300, 0)
	browser.set_window_size(1300, 400)
	browser.get('file:///home/usaid/Downloads/t-rex-runner/index.html')
	browser.execute_script("cont = document.getElementById('main-frame-error'); cont.style.transform='scale(2.1)';cont.style.top = '300px'")
except:
	print("UGH")
