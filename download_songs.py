import youtube_dl


list_top = [
    'https://www.youtube.com/watch?v=ojeb8_kkWGE',
    'https://www.youtube.com/watch?v=im9XuJJXylw',
    'https://www.youtube.com/watch?v=6Whgn_iE5uc',
    'https://www.youtube.com/watch?v=h8iPUK0AGRo',
    'https://www.youtube.com/watch?v=OPf0YbXqDm0',
    'https://www.youtube.com/watch?v=MUFasKZcH_c',
    'https://www.youtube.com/watch?v=KQ6zr6kCPj8'
]

list_average = [
    'https://www.youtube.com/watch?v=VJHJAkhacGU',
    'https://www.youtube.com/watch?v=ccenFp_3kq8',
    'https://www.youtube.com/watch?v=nxvlKp-76io',
    'https://www.youtube.com/watch?v=K1b8AhIsSYQ',
    'https://www.youtube.com/watch?v=h6uWX5WZpL0',
    'https://www.youtube.com/watch?v=GHS8hj4TdT8',
    'https://www.youtube.com/watch?v=UfmkgQRmmeE'
]

list_validate = [
    'https://www.youtube.com/watch?v=uSD4vsh1zDA',  # top
    'https://www.youtube.com/watch?v=JGwWNGJdvx8',  # top
    'https://www.youtube.com/watch?v=SybgWaQy7_c',  # average
    'https://www.youtube.com/watch?v=SmPMMitJDYg'  # average
]

if __name__ == '__main__':

    """
    TOP_SONGS = 'https://top40weekly.com/top-100-songs-of-all-time/'
    AVERAGE_SONGS = 'https://www.npr.org/sections/allsongs/2016/08/23/490961999/all' \
                    '-songs-rewind-the-worst-songs-of-all-time?t=1615481607593'
    """

    SAVE_PATH = r'C:\Users\PC\PycharmProjects\MusicAI\data'


    def download(list, subfolder):
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': SAVE_PATH + subfolder + '/%(title)s.%(ext)s',
        }

        for i, link in enumerate(list):

            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([link])


    download(list_top, r'/songs')
    # download(list_top, r'/TOP')
    # download(list_average, r'/AVERAGE')
    # download(list_validate, r'/VALIDATE')
