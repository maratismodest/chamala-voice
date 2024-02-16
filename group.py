# V3
import os
import torch
import torchaudio

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/tt/v3_tt.pt', local_file)

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)


#җ
#ү
#ә
#һ
#ң
#ө

sample_rate = 48000
speaker='dilyara'
put_accent=True

fruits = [
  "чыгыш",
  "егет",
  "фикер",
  "гаилә",
  "хезмәт",
  "хуҗалык",
  "идарә",
  "өй",
  "шәһәр",
  "су",
  "сум",
  "укучы",
  "ул",
  "урын",
  "вакыт",
  "ярар",
  "ярдәм",
  "үзәк",
  "оешма",
  "ай",
  "акча",
  "ара",
  "баш",
  "белем",
  "дин",
  "әни",
  "фән",
  "мәгълүмат",
  "өлкә",
  "рәвеш",
  "хәл",
  "яз",
  "иҗат",
  "кыз",
  "тормыш",
  "урынбасар",
  "җитәкче",
  "тау",
  "чор",
  "чара",
  "дәрәҗә",
  "әсәр",
  "ис",
  "дөнья",
  "исем",
  "көн",
  "максат",
  "очрак",
  "мөмкинлек",
  "рәис",
  "сан",
  "сугыш",
  "сорау",
  "үсеш",
  "яшь",
  "даруханә",
  "күл",
  "бала",
  "нәтиҗә",
  "гамәл",
  "дәүләт",
  "эш",
  "мөмкин",
  "хастаханә",
  "бәби",
  "эне",
  "әби",
  "туганнар",
  "кеше",
  "мәсьәлә",
  "туган",
  "чиркәү",
  "китапханә",
  "әти",
  "малай",
  "апа",
  "абый",
  "эч",
  "бәйрәм",
  "халык",
  "кибет",
  "инглиз теле",
  "сүз",
  "барырга",
  "аңларга",
  "сорарга",
  "белергә",
  "дус",
  "начар",
  "кайтырга",
  "тыңларга",
  "сәгать",
  "татар теле",
  "укырга",
  "инглизчә",
  "рус теле",
  "русча",
  "кара",
  "авыл",
  "хатын",
  "мәдәният",
  "җыр",
  "бөек",
  "мөмкинме?",
  "ил",
  "исәп",
  "китап",
  "мәчет",
  "дәрес",
  "уйнарга",
  "әйтергә",
  "алырга",
  "сатарга",
  "җавап бирергә",
  "очрашырга",
  "бина",
  "борынгы",
  "пычрак",
  "чиста",
  "биек",
  "тәбәнәк",
  "матур",
  "ямьсез",
  "тәмле",
  "музейлар",
  "район",
  "болын",
  "табигать",
  "урман",
  "бакча",
  "киң",
  "тар",
  "кечкенә",
  "кайнар",
  "файдалы",
  "файдасыз",
  "тәмсез",
  "ерак",
  "елга",
  "баллы",
  "урам",
  "йорт, өй",
  "йомшак",
  "каты",
  "иртә",
  "кич",
  "төн",
  "кунак",
  "иске",
  "яңа",
  "кат",
  "бүлмә",
  "өй жиһазы",
  "суыткыч",
  "кер юу машинасы",
  "тәрәзә",
  "якын",
  "тирән",
  "кунак бүлмәсе",
  "сай",
  "аш бүлмәсе",
  "баскыч",
  "балалар бүлмәсе",
  "юыну бүлмәсе",
  "өстәл",
  "урындык",
  "шкаф",
  "тузан суыргыч",
  "мендәр",
  "карават",
  "юрган",
  "палас",
  "якты",
  "уңайлы",
  "анда",
  "монда",
  "җылы",
  "яңгыр",
  "җил",
  "кар",
  "мәхәббәт",
  "иртәгә",
  "кичә",
  "һава торышы",
  "җәй",
  "яшел",
  "соры",
  "шәмәхә",
  "бер",
  "ике",
  "өч",
  "дүрт",
  "биш",
  "алты",
  "сигез",
  "тугыз",
  "ун",
  "чәчәк",
  "балык",
  "ит",
  "тавык",
  "кош",
  "үсемлек",
  "кием",
  "чалбар",
  "башлык, баш киеме",
  "ак",
  "сары",
  "алсу",
  "кул",
  "күз",
  "аяк киеме",
  "өс киеме",
  "балалар киеме",
  "бизәнү әйберләре",
  "уенчык",
  "итәк",
  "болыт",
  "бүген",
  "һава торышы",
  "көз",
  "кыш",
  "кызыл",
  "тән",
  "бармак",
  "колак",
  "борын",
  "авыз",
  "бит",
  "арка",
  "чәч, чәчләр",
  "тел",
  "эчемлек",
  "икмәк, ипи",
  "май",
  "сөт",
  "күкәй, йомырка",
  "чәй",
  "бал",
  "җилэк-җимеш",
  "яшелчә",
  "кишер",
  "кыяр",
  "кәбестә",
  "алма",
  "аш",
  "ботка",
  "тоз",
  "ташлама",
  "кыйбат",
  "бәя",
  "аю",
  "төлке",
  "куян",
  "сәлам",
  "хезмәткәр",
  "юл",
  "көч",
  "мәгариф",
  "мәктәп",
  "мөселман",
  "очрашу",
  "өлеш",
  "шигырь",
  "тараф",
  "тарих",
  "укытучы",
  "башкала",
  "сыер",
  "ат",
  "дуслык",
  "өмет",
  "вәкил",
  "җир",
  "түләргә",
  "эшләргә",
  "янында",
  "сөйләшергә",
  "зур",
  "салкын",
  "фатир",
  "тукталыш",
  "ишек",
  "йокы бүлмәсе",
  "салкын",
  "бүген",
  "зәңгәр",
  "көрән",
  "җиде",
  "агач",
  "хайван",
  "күлмәк",
  "теш",
  "ашамлык",
  "бәлеш",
  "бәрәңге",
  "шикәр",
  "арзан",
  "бүре",
  "эт",
  "песи",
  "хөрмәт",
  "татарча",
  "бабай",
  "кыз",
  "сеңел",
  "балалар бакчасы"
]

for x in fruits:
  print(x);
  example_text = x;
  text=example_text;
  filename=x+'.mp3'
  audio_paths = model.apply_tts(text=text+'.',
                        speaker=speaker,
                        sample_rate=sample_rate,
                        put_accent=put_accent,
                        );
  torchaudio.save('audio/' + filename,
                  audio_paths.unsqueeze(0),
                  sample_rate=sample_rate)
