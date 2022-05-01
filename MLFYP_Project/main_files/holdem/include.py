## ******************************* HAND STRENGTH METRICS ********************************
## The HandStrength() functions returns a value which helps us understand how strong our
## hand is against all of the potential hands our opponent could hold. This is useful 
## for post-flop situations when we are not sure what cards (public or private) give us a
## decent hand. Values that follow are used to compare against a player's own perceived 
## hand value.

## Author: Gary Harney

# format... {action: {raises_i_face: {bot_position: { 
hand_strength_preflop = {"betting": 
                        {
                            0: #facing_raises_debt (in this round)
                                {
                                    0: 0.25,   # Dealer
                                    1 : 0.55,   # SB
                                    2 : 0.45    # BB
                                }
                            ,
                            1: #facing_raises_debt (in this round)
                                {
                                    0: 0.32,   # Dealer
                                    1 : 0.60,   # SB
                                    2 : 0.50    # BB
                                }
                            ,
                            2: #facing_raises_debt (in this round)
                                {
                                    0: 0.55,
                                    1 : 0.70,
                                    2 : 0.66
                                }
                            
                            # cannot face more than 2 raises
                        },
                    "calling": # more like to open range wider for calling having already seen all players just checking/calling before
                        {
                            0: #facing_raises_debt (in this round)
                                {
                                    0: 0.23,   # Dealer
                                    1 : 0.48,   # SB
                                    2 : 0.40    # BB
                                }
                            ,
                            1: #facing_raises_debt (in this round)
                                {
                                    0: 0.30,   # Dealer
                                    1 : 0.55,   # SB
                                    2 : 0.47    # BB
                                }
                            ,
                            2: #facing_raises_debt (in this round)
                                {
                                    0: 0.50,
                                    1 : 0.65,
                                    2 : 0.50
                                }
                        }
                    }

hand_strength_flop = {"betting": 
                        {
                            0: #facing_raises_debt (in this round)
                                {
                                    0: 0.50,   # Dealer
                                    1 : 0.65,   # SB
                                    2 : 0.58    # BB
                                }
                            ,
                            1: #facing_raises_debt (in this round)
                                {
                                    0: 0.65,   # Dealer
                                    1 : 0.79,   # SB
                                    2 : 0.73    # BB
                                }
                            ,
                            2: #facing_raises_debt (in this round)
                                {
                                    0: 0.75,
                                    1 : 0.89,
                                    2 : 0.83
                                }
                            
                            # cannot face more than 2 raises
                        },
                    "calling": # more like to open range wider for calling having already seen all players just checking/calling before
                        {
                            0: #facing_raises_debt (in this round)
                                {
                                    0: 0.35,   # Dealer
                                    1 : 0.52,   # SB
                                    2 : 0.47    # BB
                                }
                            ,
                            1: #facing_raises_debt (in this round)
                                {
                                    0: 0.50,   # Dealer
                                    1 : 0.65,   # SB
                                    2 : 0.58    # BB
                                }
                            ,
                            2: #facing_raises_debt (in this round)
                                {
                                    0: 0.65,   
                                    1 : 0.79,   
                                    2 : 0.73    
                                }
                        }
                    }

hand_strength_turn = {"betting": 
                        {
                            0: #facing_raises_debt (in this round)
                                {
                                    0: 0.53,   # Dealer
                                    1 : 0.68,   # SB
                                    2 : 0.61    # BB
                                }
                            ,
                            1: #facing_raises_debt (in this round)
                                {
                                    0: 0.68,   # Dealer
                                    1 : 0.82,   # SB
                                    2 : 0.76    # BB
                                }
                            ,
                            2: #facing_raises_debt (in this round)
                                {
                                    0: 0.78,
                                    1 : 0.87,
                                    2 : 0.83
                                }
                            
                            # cannot face more than 2 raises
                        },
                    "calling": # more like to open range wider for calling having already seen all players just checking/calling before
                        {
                            0: #facing_raises_debt (in this round)
                                {
                                    0: 0.36,   # Dealer
                                    1 : 0.52,   # SB
                                    2 : 0.48    # BB
                                }
                            ,
                            1: #facing_raises_debt (in this round)
                                {
                                    0: 0.50,   # Dealer
                                    1 : 0.60,   # SB
                                    2 : 0.55    # BB
                                }
                            ,
                            2: #facing_raises_debt (in this round)
                                {
                                    0: 0.60,   
                                    1 : 0.75,   
                                    2 : 0.68    
                                }
                        }
                    }

hand_strength_river = {"betting": 
                        {
                            0: #facing_raises_debt (in this round)
                                {
                                    0: 0.53,   # Dealer
                                    1 : 0.68,   # SB
                                    2 : 0.61    # BB
                                }
                            ,
                            1: #facing_raises_debt (in this round)
                                {
                                    0: 0.68,   # Dealer
                                    1 : 0.82,   # SB
                                    2 : 0.76    # BB
                                }
                            ,
                            2: #facing_raises_debt (in this round)
                                {
                                    0: 0.78,
                                    1 : 0.87,
                                    2 : 0.83
                                }
                            
                            # cannot face more than 2 raises
                        },
                    "calling": # more like to open range wider for calling having already seen all players just checking/calling before
                        {
                            0: #facing_raises_debt (in this round)
                                {
                                    0: 0.36,   # Dealer
                                    1 : 0.52,   # SB
                                    2 : 0.48    # BB
                                }
                            ,
                            1: #facing_raises_debt (in this round)
                                {
                                    0: 0.50,   # Dealer
                                    1 : 0.60,   # SB
                                    2 : 0.55    # BB
                                }
                            ,
                            2: #facing_raises_debt (in this round)
                                {
                                    0: 0.60,   
                                    1 : 0.75,   
                                    2 : 0.68    
                                }
                        }
                    }



# UPPER LIMIT OF RANGE IS 7462 **
#preflop ranges (Dealer moves last)

preflop_range = {"betting": 
                    {
                        0: #facing_raises_debt (in this round)
                            {
                                # bot_position: evaluation
                                0: 6000,
                                1 : 5000,
                                2 : 5500
                            }
                        ,
                        1: #facing_raises_debt (in this round)
                            {
                                0: 6000, 
                                1 : 4000,
                                2 : 5000
                            }
                        ,
                        2: #facing_raises_debt (in this round)
                            {
                                0: 5000,
                                1 : 3500,
                                2 : 4200
                            }
                        
                        # cannot face more than 2 raises
                    },
                 "calling": # more like to open range wider for calling having already seen all players just checking/calling before
                    {
                        0: 
                            {
                                0: 8500,
                                1 : 7000,
                                2 : 7700
                            }
                        ,
                        1:
                            {
                                0: 8000, 
                                1 : 6700,
                                2 : 7500
                            }
                        ,
                        2: 
                            {
                                0: 7000, 
                                1 : 4500,
                                2 : 5200
                            }
                        
                        # cannot face more than 2 raises
                    }
                }

flop_range = {"betting": 
                    {
                        0: #facing_raises_debt (in this round)
                            {
                                0: 5600,
                                1 : 5000,
                                2 : 5300
                            }
                        ,
                        1: #facing_raises_debt (in this round)
                            {
                                0: 5300, 
                                1 : 4600,
                                2 : 5000
                            }
                        ,
                        2: #facing_raises_debt (in this round)
                            {
                                0: 5000,
                                1 : 3500,
                                2 : 4200
                            }
                        
                        # cannot face more than 2 raises
                    },
                 "calling": # more like to open range wider for calling having already seen all players just checking/calling before
                    {
                        0: 
                            {
                                0: 8400,
                                1 : 7000,
                                2 : 7700
                            }
                        ,
                        1:
                            {
                                0: 8000, 
                                1 : 6700,
                                2 : 7500
                            }
                        ,
                        2: 
                            {
                                0: 7000, 
                                1 : 4500,
                                2 : 5200
                            }
                        
                        # cannot face more than 2 raises
                    }
                }

turn_range = {"betting": 
                    {
                        0: #facing_raises_debt (in this round)
                            {
                                0: 5000,
                                1 : 3000,
                                2 : 4000
                            }
                        ,
                        1: #facing_raises_debt (in this round)
                            {
                                0: 4000,
                                1 : 2000,
                                2 : 3000
                            }
                        ,
                        2: #facing_raises_debt (in this round)
                            {
                                0: 3500,
                                1 : 2500,
                                2 : 3000
                            }
                        
                        # cannot face more than 2 raises
                    },
                 "calling": # more like to open range wider for calling having already seen all players just checking/calling before
                    {
                        0: 
                            {
                                0: 8500,
                                1 : 7000,
                                2 : 7700
                            }
                        ,
                        1:
                            {
                                0: 8000, 
                                1 : 6700,
                                2 : 7500
                            }
                        ,
                        2: 
                            {
                                0: 7000, 
                                1 : 4500,
                                2 : 5200
                            }
                        
                        # cannot face more than 2 raises
                    }
                }


river_range = {"betting": 
                    {
                        0: #facing_raises_debt (in this round)
                            {
                                0: 5000,
                                1 : 3000,
                                2 : 4000
                            }
                        ,
                        1: #facing_raises_debt (in this round)
                            {
                                0: 4000,
                                1 : 2000,
                                2 : 3000
                            }
                        ,
                        2: #facing_raises_debt (in this round)
                            {
                                0: 3500,
                                1 : 2500,
                                2 : 3000
                            }
                        
                        # cannot face more than 2 raises
                    },
                 "calling": # more like to open range wider for calling having already seen all players just checking/calling before
                    {
                        0: 
                            {
                                0: 8500,
                                1 : 7000,
                                2 : 7700
                            }
                        ,
                        1:
                            {
                                0: 8000, 
                                1 : 6700,
                                2 : 7500
                            }
                        ,
                        2: 
                            {
                                0: 7000, 
                                1 : 4500,
                                2 : 5200
                            }
                        
                        # cannot face more than 2 raises
                    }
                }