import mysql.connector



class Model:

    def __init__(self):
        self.song_dict = {}
        self.db_status = True
        self.conn = None
        self.cur = None
        try:
            self.conn = mysql.connector.connect(host="localhost", user="root", passwd="12345", database="student_detail")
            print("databaseconnectd succesfully")
            self.cur = self.conn.cursor()
        except:
            self.db_status = False


    def get_db_status(self):
        return self.db_status

    def close_db_connection(self):
        if self.cur is not None:
            self.cur.close()
            print("Cursor closed successfully")
        if self.conn is not None:
            self.conn.close()
            print("Disconnected successfully from the DB")
    def insert_Id_In_database(self,id,name,emailid,gender,branch):
        is_Id_present = self.search_Id_In_database(id)
        if (is_Id_present):
            return "Id Is already Present "
        else:
            # self.cur.execute("select max(song_id) from myfavourites")
            # last_song_id=self.cur.fetchone()[0]
            # next_song_id=last_song_id+1
            # if last_song_id is not None:
            #   next_song_id=last_song_id+1
            self.cur.execute("insert into info values(%s,%s,%s,%s,%s)", (id, name,emailid,gender,branch))
            self.conn.commit()
            return "Sucessfully Submit"

    def search_Id_In_database(self,id):
        Id = (id,)
        self.cur.execute("select id from info where Id= %s ", Id)
        id_tuple = self.cur.fetchone()
        if (id_tuple is None):
            return False
        else:
            return True
    def remove_Id_In_database(self,Id):
        id = (Id,)
        self.cur.execute("delete from info where Id= %s", id)
        self.conn.commit()
        return "Id removed from info"
    def load_Id_In_database(self):
        self.cur.execute("select id,name,email,gender,branch from info")
        song_present = False
        for x, y in self.cur:
            self.song_dict[x] = y
            song_present = True
        if song_present == True:
            return "List populated from favourite"
        else:
            return "No song present in your favourites"




    """
    def add_song(self, song_name, song_path):
        self.song_dict[song_name] = song_path
        print("song added:", self.song_dict[song_name])

    def get_song_path(self, song_name):
        return self.song_dict[song_name]

    def remove_song(self, song_name):
        self.song_dict.pop(song_name)
        print(self.song_dict)
    def search_song_in_favourites(self,song_name):
        song=(song_name,)
        self.cur.execute("select song_name from myfavourites where song_name= %s ",song)
        song_tuple=self.cur.fetchone()
        if(song_tuple is None):
            return False
        else:
            return True
    def add_song_to_favourites(self,song_name,song_path):
        is_song_present=self.search_song_in_favourites(song_name)
        if( is_song_present):
            return "song is already present"
        else:
            #self.cur.execute("select max(song_id) from myfavourites")
            #last_song_id=self.cur.fetchone()[0]
            #next_song_id=last_song_id+1
            #if last_song_id is not None:
             #   next_song_id=last_song_id+1
            self.cur.execute("insert into myfavourites values(%s,%s)",(song_name,song_path))
            self.conn.commit()
            return "song added to your favourites"

    def load_song_from_favourites(self):
        self.cur.execute("select song_name,song_path from myfavourites")
        song_present=False
        for x,y in self.cur:
            self.song_dict[x]=y
            song_present=True
        if song_present==True:
            return "List populated from favourite"
        else:
            return "No song present in your favourites"
    def remove_song_from_favourites(self,song_name):
        song=(song_name,)
        self.cur.execute("delete from myfavourites where song_name= %s",song)
        self.conn.commit()
        return "song removed from favourites"
"""

if(__name__=="__main__"):
    obj = Model()