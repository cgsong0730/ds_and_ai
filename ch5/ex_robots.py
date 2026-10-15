from urllib import robotparser

rp = robotparser.RobotFileParser()
rp.parse([
    'User-agent: *',
    'Disallow: /admin/',
    'Crawl-delay: 2',
])

print(rp.can_fetch('*', 'https://example.com/books/1'))
print(rp.can_fetch('*', 'https://example.com/admin/'))
print(rp.crawl_delay('*'))
