from moviepy.editor import VideoFileClip
# from IPython.display import HTML
from pipeline import pipeline

write_output = 'project_output_2.mp4'
clip1 = VideoFileClip('project_video.mp4')
write_clip = clip1.fl_image(pipeline)
write_clip.write_videofile(write_output, audio=False)