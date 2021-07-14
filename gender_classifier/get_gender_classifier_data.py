#!/usr/bin/python
DJANGO_SETTINGS_MODULE = 'vapp.settings'
from vapp import settings
from django.core.management  import setup_environ

setup_environ( settings )
import sys, argparse, uuid
sys.path.append( '/usr/local/voicesite/vapp' )
sys.path.append( '/usr/local/voicesite/' )

from mnews.models import *
from vapp.utils import generate_attachment, send_email
from log import get_request_logger
from app_manager.models import App_instance
from vapp.utils import *
from dateutil.rrule import rrule, DAILY

logger = get_request_logger()

parser = argparse.ArgumentParser()
#parser.add_argument( "--ai_id", help = "ai_id for generating data, ex:'10'")
parser.add_argument( "--thresh", help = "The number of items required for training in each class", required=True)
parser.add_argument( "--start_date", help = "Start date YYYY-MM-DD", required=True)
parser.add_argument( "--end_date", help = "End date YYYY-MM-DD", required=True)
parser.add_argument("--email_ids", help="Recipient email_ids (comma separated)", required=True)
parser.add_argument("--output", help="Path where worksheet needs to be generated", required=True)

args = parser.parse_args()

email_ids = [email_id.strip() for email_id in args.email_ids.split(',')]
output_file = args.output

if args.start_date and args.end_date is not None:
    start_datetime = datetime.strptime( args.start_date, '%Y-%m-%d' ).date()
    end_datetime = datetime.strptime( args.end_date, '%Y-%m-%d' ).date()
    news_males = News.objects.filter(time__gte = start_datetime, time__lt = end_datetime, state__in = ['ARC', 'PUB'], source__in=[1,2], gender_id=1)
    news_females = News.objects.filter(time__gte = start_datetime, time__lt = end_datetime, state__in = ['ARC', 'PUB'], source__in=[1,2], gender_id=2)
    #news = News.objects.filter(time__gte = start_datetime, time__lt = end_datetime, state__in = ['ARC', 'PUB'])

    if news_males.count() < int(args.thresh):
        print("Lesser than threshold number of items in Male class, please increase the dates because there are {} items".format(news_males.count()))
        sys.exit() 
    if news_females.count() < int(args.thresh):
        print("Lesser than threshold number of items in Female class, please increase the dates because there are {} items".format(news_females.count()))
        sys.exit() 

def item_detail (news, email_ids, output_file):

    data = [['Instance name', 'Unique Item ID', 'Title', 'CallerId','State', 'Block', 'District', 'Item created date', 'transcription', 'state','tags','Recording audio link', 'Total listening duration', 'Item duration', 'published date', 'Format', 'Gender', 'Checksum']]#, 'ML transcript']]#, 'Occupation', 'Is Comment']]#, 'ML transcript','ML State', 'ML Gender', 'ML tags',]]
    for idx, item in enumerate(news):
      if idx >= int(args.thresh):
          break
      d = []
      d.append(item.ai.name)
      d.append(item.id)
      d.append(item.title)
      d.append(item.callerid)
      if item.location:
         d.append(item.location.state.name)
         d.append(item.location.block.name)
         d.append(item.location.district.name)
      else:
         d.append('NA')
         d.append('NA')
         d.append('NA')
      d.append(item.time)
      d.append(item.transcript)
      d.append(item.state)
      d.append(item.tags)
      d.append("http://voice.gramvaani.org/fsmedia/recordings/" + str(item.ai_id) + '/' + str(item.detail_id) + '.mp3')
      if item.id: 
          ihs = Item_heard_stats.objects.filter(item = item).values_list ('duration', flat = True)
          d.append(sum(ihs))
      else:
          d.append('NA') 

      if item.detail:
          d.append(item.get_news_duration())
      else:
          d.append('Audio Not found')

      if item.state in ['PUB', 'ARC']:
        me = ModerationEventRecorder.objects.filter(item = item, event_type = 'ITEM_PUBLISHED')
        if me:
            d.append(me[0].timestamp)
        else:
            d.append('Not Published')
      else:
        d.append('NA')

      if item.format_id:
          d.append(item.format_id)
      else:
          d.append('NA')

      if item.gender_id:
          d.append(item.gender_id)
      else:
          d.append('NA')

      if item.checksum:
          d.append(item.checksum)
      else:
          d.append('NA')

      #if item.age_group:
      #    d.append(item.age_group)
      #else:
      #    d.append('NA')

      #if item.occupation:
      #    d.append(item.occupation)
      #else:
      #    d.append('NA')

      #if item.is_comment:
      #    d.append(item.is_comment)
      #else:
      #    d.append('NA')

      #if item.is_autopublished:
      #    d.append(item.is_autopublished)
      #else:
      #    d.append('NA')

      #if item.ml_properties:
          #try:
              #dic = json.loads(item.ml_properties)
              #mstate = dic.get('ml_state', '')
              #mgender = dic.get('ml_gender', '')
              #mtags = dic.get('ml_tags', '')
              #mtranscript = dic.get('ml_transcript', '')
              #d.append(mstate)
              #d.append(mgender)
              #d.append(mtags)
              #d.append(mtranscript)
          #except Exception as e:
              #d.append('')
      #else:
          #d.append('')

      data.append(d)

    return data
    #generate_attachment(data, output_file)
    #send_email('Tag Data', 'Tag Data', email_ids, output_file)


try:
    data_males = item_detail(news_males, email_ids, output_file)
    data_males = data_males[1:]    # Removing the header repetition
    data_females = item_detail(news_females, email_ids, output_file)

    data_females.extend(data_males)
     
    generate_attachment(data_females, output_file)
    send_email('Tag Data', 'Tag Data', email_ids, output_file)
except Exception as e:
     logger.exception("could not generate campaign task %s" % ( e ))

