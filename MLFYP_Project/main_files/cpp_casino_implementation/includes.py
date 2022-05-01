limit = 2

# UPPER LIMIT OF RANGE IS 7462 **


#preflop ranges (Dealer moves last)
# {action: {facing_raises_debt: {bot_position: { 
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

turn_river = {"betting": 
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



