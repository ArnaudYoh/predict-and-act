'''
ViZDoom wrapper
'''
from __future__ import print_function
import sys
import os

# vizdoom_path = '../../../../toolboxes/ViZDoom_2017_03_31'
vizdoom_path = '/home/naunauyoh/anaconda3/lib/python3.7/site-packages/vizdoom'
sys.path = [os.path.join(vizdoom_path,'bin/python3')] + sys.path

import vizdoom 
print(vizdoom.__file__)
import random
import time
import numpy as np
import re
import cv2

class DoomSimulator:
    
    def __init__(self, args):        
        self.config = args['config']
        self.resolution = args['resolution']
        self.frame_skip = args['frame_skip']
        self.color_mode = args['color_mode']
        self.switch_maps = args['switch_maps']
        self.maps = args['maps']
        self.game_args = args['game_args']
        
        self._game = vizdoom.DoomGame()
        self._game.add_available_game_variable(vizdoom.GameVariable.POSITION_X)
        self._game.add_available_game_variable(vizdoom.GameVariable.POSITION_Y)
        self._game.add_available_game_variable(vizdoom.GameVariable.POSITION_Z)
        self._game.set_objects_info_enabled(True)
        self._game.set_sectors_info_enabled(True)
        self._game.set_vizdoom_path(os.path.join(vizdoom_path,'vizdoom'))
        self._game.set_doom_game_path(os.path.join(vizdoom_path,'freedoom2.wad'))
        self._game.load_config(self.config)
        self._game.add_game_args(self.game_args)
        self.curr_map = 0
        self._game.set_doom_map(self.maps[self.curr_map])
        
        # set resolution
        try:
            self._game.set_screen_resolution(getattr(vizdoom.ScreenResolution, 'RES_%dX%d' % self.resolution))
            self.resize = False
        except:
            print("Requested resolution not supported:", sys.exc_info()[0], ". Setting to 160x120 and resizing")
            self._game.set_screen_resolution(getattr(vizdoom.ScreenResolution, 'RES_160X120'))
            self.resize = True

        # set color mode
        if self.color_mode == 'RGB':
            self._game.set_screen_format(vizdoom.ScreenFormat.CRCGCB)
            self.num_channels = 3
        elif self.color_mode == 'GRAY':
            self._game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
            self.num_channels = 1
        else:
            print("Unknown color mode")
            raise

        self.available_controls, self.continuous_controls, self.discrete_controls = self.analyze_controls(self.config)
        self.num_buttons = self._game.get_available_buttons_size()
        assert(self.num_buttons == len(self.discrete_controls) + len(self.continuous_controls))
        assert(len(self.continuous_controls) == 0) # only discrete for now
        self.num_meas = self._game.get_available_game_variables_size()
        self.num_objects = 4
            
        self.meas_tags = []
        for nm in range(self.num_meas):
            self.meas_tags.append('meas' + str(nm))
            
        self.episode_count = 0
        self.game_initialized = False
        
    def analyze_controls(self, config_file):
        with open(config_file, 'r') as myfile:
            config = myfile.read()
        m = re.search('available_buttons[\s]*\=[\s]*\{([^\}]*)\}', config)
        avail_controls = m.group(1).split()
        cont_controls = np.array([bool(re.match('.*_DELTA', c)) for c in avail_controls])
        discr_controls = np.invert(cont_controls)
        return avail_controls, np.squeeze(np.nonzero(cont_controls)), np.squeeze(np.nonzero(discr_controls))
        
    def init_game(self):
        if not self.game_initialized:
            self._game.init()
            self.game_initialized = True
            
    def close_game(self):
        if self.game_initialized:
            self._game.close()
            self.game_initialized = False

    def get_closest(self, object_poss, player_pos):
        min_idx = None
        min_dist = None
        if not object_poss:
            return [0, 0]
        for idx, object_pos in enumerate(object_poss):
            curr_dist = np.sqrt((object_pos[0] - player_pos[0])**2 + (object_pos[1] - player_pos[1])**2)
            if min_idx is None or min_dist > curr_dist:
                min_dist = curr_dist
                min_idx = idx

        return [object_poss[min_idx][0] - player_pos[0], object_poss[min_idx][1] - player_pos[1]]
            
    def step(self, action=0):
        """
        Action can be either the number of action or the actual list defining the action
        
        Args:
            action - action encoded either as an int (index of the action) or as a bool vector
        Returns:
            img  - image after the step
            meas - numpy array of returned additional measurements (e.g. health, ammo) after the step
            rwrd - reward after the step
            term - if the state after the step is terminal
        """
        self.init_game()
        
        rwrd = self._game.make_action(action, self.frame_skip)
        state = self._game.get_state()
        
        if state is None:
            img = None
            meas = None
            game_object = None
        else:        
            # ViZDoom 1.0
            #raw_img = state.image_buffer
                
            ## ViZDoom 1.1 
            if self.color_mode == 'RGB':
                raw_img = state.screen_buffer
            elif self.color_mode == 'GRAY':
                raw_img = np.expand_dims(state.screen_buffer,0)
                
            if self.resize:
                if self.num_channels == 1:
                    if raw_img is None or (isinstance(raw_img, list) and raw_img[0] is None):
                        img = None
                    else:
                        img = cv2.resize(raw_img[0], (self.resolution[0], self.resolution[1]))[None,:,:]
                else:
                    raise NotImplementedError('not implemented for non-Grayscale images')
            else:
                img = raw_img
                
            meas = state.game_variables # this is a numpy array of game variables specified by the scenario
            health_kits_pos = []
            poison_vial_pos = []
            player_pos = None
            for o in state.objects:
                if o.name == 'DoomPlayer':
                    player_pos = (o.position_x, o.position_y)
                    break

            for label in state.labels:
                if label.name == 'CustomMedikit' or label.name == 'Medikit':
                    health_kits_pos.append((label.object_position_x, label.object_position_y))
                elif label.name == 'Poison':
                    poison_vial_pos.append((label.object_position_x, label.object_position_y))

            closest_health = self.get_closest(health_kits_pos, player_pos)
            closest_poison = self.get_closest(poison_vial_pos, player_pos)

            game_object = np.array([closest_health, closest_poison])
            game_object = np.concatenate(game_object)
            
        term = self._game.is_episode_finished() or self._game.is_player_dead()
        
        if term:
            self.new_episode() # in multiplayer multi_simulator takes care of this            
            img = np.zeros((self.num_channels, self.resolution[1], self.resolution[0]), dtype=np.uint8) # should ideally put nan here, but since it's an int...
            meas = np.zeros(self.num_meas, dtype=np.uint32) # should ideally put nan here, but since it's an int...
            game_object = np.zeros((self.num_objects,))
            
        return img, meas, game_object, rwrd, term
    
    def get_random_action(self):
        return [(random.random() >= .5) for i in range(self.num_buttons)]
        
    def is_new_episode(self):
        return self._game.is_new_episode()
    
    def next_map(self):     
        if self.switch_maps:
            self.curr_map = (self.curr_map+1) % len(self.maps)
            self._game.set_doom_map(self.maps[self.curr_map])
    
    def new_episode(self):
        self.next_map()
        self.episode_count += 1
        self._game.new_episode()
